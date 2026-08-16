"""
ZARI.ai — Full Backbone Unfreezing & Enhanced Field Augmentation Pipeline (train_full_model.py)

This script trains a completely fresh model from pretrained ImageNet-21k weights:
- Full backbone unfreezing (tf_efficientnetv2_s.in21k_ft_in1k)
- Enhanced Field Augmentation (384x384 resolution, rotation, affine, jitter, perspective, blur)
- Evidential Deep Learning (EDL) Loss with kl-annealing over 10 steps
- Differential AdamW learning rates (1e-5 backbone, 3e-4 EDL head)
- CosineAnnealingLR scheduler
- Early stopping (patience=2) based on field validation accuracy
- Comprehensive evaluation on field test set (6,648 images)

Artifacts produced:
- Model Checkpoint : ml_pipeline/models/full_model.pth
- Training History : ml_pipeline/logs/full_model_history.json
- Test Evaluation  : ml_pipeline/ANALYSIS_COMPLETE/reports/full_model_report.txt
"""

import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import f1_score, roc_auc_score
import timm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm


# =============================================================================
# 1. GPU & ENVIRONMENT CHECK
# =============================================================================
assert torch.cuda.is_available(), "CUDA is required for training full_model!"
device = torch.device("cuda")
print("===========================================================================")
print("  ZARI.ai — FULL BACKBONE UNFREEZING & ENHANCED FIELD AUGMENTATION")
print("===========================================================================")
print(f"Using compute device: {device}")
print(f"GPU Name: {torch.cuda.get_device_name(0)}")
print(f"VRAM Available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")


# =============================================================================
# 2. LOAD DATASET
# =============================================================================
DATASET_CSV = "ml_pipeline/data/dataset_final_training.csv"
df = pd.read_csv(DATASET_CSV)

# Field-only training split (PlantCity & NWRD)
train_df = df[(df["split"] == "train") & (df["source_dataset"].isin(["plantcity", "nwrd"]))].copy()
val_df = df[df["split"] == "val"].copy()
test_df = df[df["split"] == "test"].copy()

print(f"[STEP 1] Loaded dataset: {DATASET_CSV} ({len(df):,} total rows)")
print(f"✓ Field Training samples  : {len(train_df):,}")
print(f"✓ Field Validation samples: {len(val_df):,}")
print(f"✓ Field Test samples      : {len(test_df):,}")


# =============================================================================
# 3. DATASET CLASS
# =============================================================================
class PlantDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = Path(row["image_path"])
        if not img_path.exists():
            img_path = Path("ml_pipeline/data/raw") / str(row["image_path"]).split("raw/")[-1]
        img = Image.open(img_path).convert("RGB")
        label = int(row["class_id"])
        if self.transform:
            img = self.transform(img)
        return img, label


# =============================================================================
# 4. ENHANCED FIELD AUGMENTATION (384x384 RESOLUTION)
# =============================================================================
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(size=384, scale=(0.5, 1.0), ratio=(0.75, 1.33)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.3),
    transforms.RandomRotation(degrees=30),
    transforms.RandomPerspective(distortion_scale=0.3, p=0.4),
    transforms.RandomApply([transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), shear=(-20, 20))], p=0.3),
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.4, hue=0.1),
    transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 3.0)),
    transforms.RandomApply([transforms.GaussianBlur(kernel_size=7, sigma=(1.0, 2.0))], p=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# =============================================================================
# 5. DATALOADERS
# =============================================================================
train_loader = DataLoader(
    PlantDataset(train_df, train_transform),
    batch_size=32,
    shuffle=True,
    num_workers=8,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=4,
)

val_loader = DataLoader(
    PlantDataset(val_df, val_transform),
    batch_size=32,
    shuffle=False,
    num_workers=8,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=4,
)

test_loader = DataLoader(
    PlantDataset(test_df, val_transform),
    batch_size=32,
    shuffle=False,
    num_workers=8,
    pin_memory=True,
    prefetch_factor=4,
)


# =============================================================================
# 6. CLASS WEIGHTS
# =============================================================================
CLASS_WEIGHTS_PATH = "ml_pipeline/data/class_weights.json"
with open(CLASS_WEIGHTS_PATH) as f:
    weights_data = json.load(f)
head_weights = torch.tensor([weights_data["head_weights"][str(i)] for i in range(67)]).cuda()
print(f"✓ Loaded class weights from {CLASS_WEIGHTS_PATH}")


# =============================================================================
# 7. MODEL — FULLY UNFROZEN EFFICIENTNETV2-S
# =============================================================================
backbone = timm.create_model("tf_efficientnetv2_s.in21k_ft_in1k", pretrained=True)
backbone.reset_classifier(0)
backbone = backbone.cuda()

# FULL UNFREEZE
for param in backbone.parameters():
    param.requires_grad = True


class EDLHead(nn.Module):
    def __init__(self, in_features=1280, num_classes=67):
        super().__init__()
        self.num_classes = num_classes
        self.fc = nn.Linear(in_features, num_classes)
        self.softplus = nn.Softplus()

    def forward(self, x):
        logits = self.fc(x)
        evidence = self.softplus(logits)
        alpha = evidence + 1.0
        alpha = torch.clamp(alpha, min=1.0 + 1e-6)
        S = alpha.sum(dim=-1, keepdim=True)
        probability = alpha / S
        uncertainty = self.num_classes / S.squeeze(-1)
        return evidence, alpha, uncertainty, probability


edl_head = EDLHead(1280, 67).cuda()

trainable_backbone = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
trainable_head = sum(p.numel() for p in edl_head.parameters() if p.requires_grad)
print("[STEP 2] Initializing Fully Unfrozen Backbone & EDL Head...")
print(f"✓ Backbone Trainable Params: {trainable_backbone:,}")
print(f"✓ EDL Head Trainable Params: {trainable_head:,}")
print(f"✓ Total Trainable Params   : {trainable_backbone + trainable_head:,}")


# =============================================================================
# 8. EDL LOSS WITH KL ANNEALING
# =============================================================================
class EDLLoss(nn.Module):
    def __init__(self, num_classes=67, annealing_step=10):
        super().__init__()
        self.num_classes = num_classes
        self.annealing_step = annealing_step
        self.current_epoch = 0

    def kl_divergence(self, alpha, target):
        alpha_tilde = target + (1.0 - target) * alpha
        sum_alpha_tilde = torch.sum(alpha_tilde, dim=-1, keepdim=True)
        kl = (
            torch.lgamma(sum_alpha_tilde)
            - torch.lgamma(torch.tensor(float(self.num_classes), device=alpha.device))
            - torch.sum(torch.lgamma(alpha_tilde), dim=-1, keepdim=True)
            + torch.sum(
                (alpha_tilde - 1.0) * (torch.digamma(alpha_tilde) - torch.digamma(sum_alpha_tilde)),
                dim=-1,
                keepdim=True,
            )
        )
        return kl.squeeze(-1)

    def forward(self, evidence, target, class_weights=None):
        alpha = evidence + 1.0
        alpha = torch.clamp(alpha, min=1.0 + 1e-6)
        y = torch.nn.functional.one_hot(target, num_classes=self.num_classes).float()
        S = alpha.sum(dim=-1, keepdim=True)
        p = alpha / S
        loss_mse = ((y - p) ** 2).sum(dim=-1)
        loss_var = (p * (1 - p) / (S + 1)).sum(dim=-1)
        if class_weights is not None:
            weight_per_sample = class_weights[target]
            loss_mle = ((loss_mse + loss_var) * weight_per_sample).mean()
        else:
            loss_mle = (loss_mse + loss_var).mean()
        annealing_coef = min(1.0, self.current_epoch / self.annealing_step)
        kl = self.kl_divergence(alpha, y).mean()
        loss_kl = annealing_coef * kl
        return loss_mle + loss_kl

    def update_epoch(self, epoch):
        self.current_epoch = epoch


# =============================================================================
# 9. OPTIMIZER, SCHEDULER & MIXED PRECISION
# =============================================================================
criterion = EDLLoss(67, annealing_step=10)
optimizer = torch.optim.AdamW([
    {"params": backbone.parameters(), "lr": 1e-5, "weight_decay": 0.01},
    {"params": edl_head.parameters(), "lr": 3e-4, "weight_decay": 0.01},
])
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15, eta_min=1e-6)
scaler = torch.amp.GradScaler("cuda")


# =============================================================================
# 10. TRAINING LOOP WITH EARLY STOPPING
# =============================================================================
MAX_EPOCHS = 15
PATIENCE = 2
BEST_MODEL_PATH = "ml_pipeline/models/full_model.pth"
HISTORY_JSON_PATH = "ml_pipeline/logs/full_model_history.json"
REPORT_TXT_PATH = "ml_pipeline/ANALYSIS_COMPLETE/reports/full_model_report.txt"

Path("ml_pipeline/models").mkdir(parents=True, exist_ok=True)
Path("ml_pipeline/logs").mkdir(parents=True, exist_ok=True)
Path("ml_pipeline/ANALYSIS_COMPLETE/reports").mkdir(parents=True, exist_ok=True)

best_val_acc = 0.0
no_improvement = 0
history = {
    "epoch": [],
    "train_loss": [],
    "val_loss": [],
    "val_acc": [],
    "val_uncertainty": [],
    "epoch_time_seconds": [],
}

print(f"\n[STEP 3] Starting Full Model Training (Max Epochs: {MAX_EPOCHS}, Early Stopping Patience: {PATIENCE})...")

start_total_time = time.time()

for epoch in range(MAX_EPOCHS):
    criterion.update_epoch(epoch)
    epoch_start = time.time()

    # --- TRAIN ---
    backbone.train()
    edl_head.train()
    running_loss = 0.0
    accum_steps = 2
    optimizer.zero_grad(set_to_none=True)

    train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1:02d}/{MAX_EPOCHS} [Train]", leave=False)
    for step_idx, (images, labels) in enumerate(train_pbar):
        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)

        with torch.amp.autocast("cuda"):
            features = backbone(images)
            evidence, alpha, uncertainty, prob = edl_head(features)
            loss = criterion(evidence, labels, head_weights)
            loss_accum = loss / accum_steps

        scaler.scale(loss_accum).backward()

        if (step_idx + 1) % accum_steps == 0 or (step_idx + 1) == len(train_loader):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        running_loss += loss.item()
        train_pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    epoch_train_loss = running_loss / len(train_loader)

    # --- VALIDATE ---
    backbone.eval()
    edl_head.eval()
    val_loss_sum = 0.0
    val_correct = 0
    val_total = 0
    val_unc_sum = 0.0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)

            with torch.amp.autocast("cuda"):
                features = backbone(images)
                evidence, alpha, uncertainty, prob = edl_head(features)
                v_loss = criterion(evidence, labels, head_weights)

            predictions = prob.argmax(dim=1)
            val_correct += (predictions == labels).sum().item()
            val_total += labels.size(0)
            val_loss_sum += v_loss.item()
            val_unc_sum += uncertainty.mean().item()

    epoch_val_loss = val_loss_sum / len(val_loader)
    epoch_val_acc = val_correct / val_total
    epoch_val_unc = val_unc_sum / len(val_loader)
    epoch_elapsed = time.time() - epoch_start

    # Record history
    history["epoch"].append(epoch + 1)
    history["train_loss"].append(round(epoch_train_loss, 4))
    history["val_loss"].append(round(epoch_val_loss, 4))
    history["val_acc"].append(round(epoch_val_acc * 100, 2))
    history["val_uncertainty"].append(round(epoch_val_unc, 4))
    history["epoch_time_seconds"].append(round(epoch_elapsed, 1))

    # --- SAVE BEST MODEL & EARLY STOPPING ---
    improved_flag = ""
    if epoch_val_acc > best_val_acc:
        best_val_acc = epoch_val_acc
        no_improvement = 0
        torch.save(
            {
                "backbone": backbone.state_dict(),
                "edl_head": edl_head.state_dict(),
                "epoch": epoch + 1,
                "val_acc": epoch_val_acc,
                "val_loss": epoch_val_loss,
                "val_uncertainty": epoch_val_unc,
            },
            BEST_MODEL_PATH,
        )
        improved_flag = "⭐ BEST"
    else:
        no_improvement += 1
        improved_flag = f"(no imp: {no_improvement}/{PATIENCE})"

    print("-" * 75)
    print(
        f"Epoch {epoch+1:02d}/{MAX_EPOCHS} | Train Loss: {epoch_train_loss:.4f} | "
        f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc*100:.2f}% | "
        f"Uncertainty: {epoch_val_unc:.4f} | Time: {epoch_elapsed:.1f}s {improved_flag}"
    )
    print("-" * 75)

    if no_improvement >= PATIENCE:
        print(f"Early stopping triggered at epoch {epoch+1}")
        break

    scheduler.step()

total_training_minutes = (time.time() - start_total_time) / 60.0
print(f"\n✓ Training completed in {total_training_minutes:.2f} minutes.")
print(f"✓ Best Field Validation Accuracy: {best_val_acc * 100:.2f}%")

with open(HISTORY_JSON_PATH, "w") as f:
    json.dump(history, f, indent=2)
print(f"✓ Saved training history to: {HISTORY_JSON_PATH}")


# =============================================================================
# 11. EVALUATE BEST MODEL ON TEST SET
# =============================================================================
print(f"\n[STEP 4] Evaluating Best Checkpoint ({BEST_MODEL_PATH}) on Field Test Set ({len(test_df):,} images)...")

checkpoint = torch.load(BEST_MODEL_PATH, map_location=device, weights_only=False)
backbone.load_state_dict(checkpoint["backbone"])
edl_head.load_state_dict(checkpoint["edl_head"])
backbone.eval()
edl_head.eval()

y_true = []
y_pred = []
y_probs = []
y_unc = []

with torch.no_grad():
    for images, labels in tqdm(test_loader, desc="Testing"):
        images = images.cuda(non_blocking=True)
        with torch.amp.autocast("cuda"):
            features = backbone(images)
            evidence, alpha, uncertainty, prob = edl_head(features)

        preds = prob.argmax(dim=1).cpu().numpy()
        probs_np = prob.cpu().numpy()
        unc_np = uncertainty.cpu().numpy()

        y_true.extend(labels.numpy())
        y_pred.extend(preds)
        y_probs.extend(probs_np)
        y_unc.extend(unc_np)

y_true = np.array(y_true)
y_pred = np.array(y_pred)
y_probs = np.array(y_probs)
y_unc = np.array(y_unc)

overall_acc = (y_pred == y_true).mean()
macro_f1 = f1_score(y_true, y_pred, average="macro")
weighted_f1 = f1_score(y_true, y_pred, average="weighted")
try:
    auroc = roc_auc_score(y_true, y_probs, multi_class="ovr")
except Exception:
    auroc = 0.0

# Class mapping
with open("ml_pipeline/data/class_to_idx.json") as f:
    class_to_idx = json.load(f)
idx_to_class = {v: k for k, v in class_to_idx.items()}

# Wheat class metrics
wheat_classes = [c for c in class_to_idx.keys() if "Wheat" in c]
wheat_metrics = {}
for w_cls in wheat_classes:
    c_idx = class_to_idx[w_cls]
    mask = y_true == c_idx
    if mask.sum() > 0:
        c_acc = (y_pred[mask] == c_idx).mean()
        wheat_metrics[w_cls] = (c_acc, mask.sum())

print("\n" + "=" * 75)
print("  ZARI.ai — FULL MODEL TEST EVALUATION RESULTS")
print("=" * 75)
print(f"Overall Test Accuracy : {overall_acc * 100:.2f}%")
print(f"Macro F1 Score        : {macro_f1:.4f}")
print(f"Weighted F1 Score     : {weighted_f1:.4f}")
print(f"AUROC (OVR)           : {auroc:.4f}")
print("-" * 75)
print("WHEAT CLASSES ACCURACY:")
for w_cls, (c_acc, count) in wheat_metrics.items():
    print(f"  - {w_cls:<25}: {c_acc * 100:.2f}% ({count} samples)")
print("=" * 75)

# Save Report TXT
report_lines = [
    "=" * 80,
    "ZARI.ai — FULL MODEL TEST EVALUATION & METRICS REPORT",
    "=" * 80,
    f"Date / Timestamp           : {pd.Timestamp.now().isoformat()}",
    f"Evaluation Dataset         : Field Test Split ({len(test_df):,} Field Images)",
    f"Evaluated Model Checkpoint : {BEST_MODEL_PATH}",
    "Training Setup             : Fully Unfrozen Backbone (tf_efficientnetv2_s.in21k_ft_in1k)",
    "                             Enhanced Field Augmentation (384x384 Resolution)",
    "                             AdamW (Backbone: 1e-5, Head: 3e-4) + CosineAnnealingLR",
    "                             EDL Head + EDLLoss with KL Annealing",
    "",
    "OVERALL METRICS:",
    "-" * 80,
    f"Overall Field Test Accuracy: {overall_acc * 100:.2f}%",
    f"Macro F1 Score             : {macro_f1:.4f}",
    f"Weighted F1 Score          : {weighted_f1:.4f}",
    f"AUROC (Multiclass OVR)     : {auroc:.4f}",
    f"Average Test Uncertainty   : {y_unc.mean():.4f}",
    "",
    "PER-CLASS PERFORMANCE (WHEAT SPECIALTY FOCUS):",
    "-" * 80,
]
for w_cls, (c_acc, count) in wheat_metrics.items():
    report_lines.append(f"{w_cls:<30}: {c_acc * 100:.2f}% (Support: {count})")

report_lines.extend([
    "",
    "=" * 80,
    "FILES CREATED & SAVED:",
    "=" * 80,
    f"✓ Model Checkpoint : {BEST_MODEL_PATH}",
    f"✓ History JSON     : {HISTORY_JSON_PATH}",
    f"✓ Evaluation Report: {REPORT_TXT_PATH}",
    "================================================================================",
])

with open(REPORT_TXT_PATH, "w") as f:
    f.write("\n".join(report_lines))

print(f"✓ Saved full evaluation report to: {REPORT_TXT_PATH}")
print("\n✅ FULL MODEL PIPELINE EXECUTION COMPLETE!")
