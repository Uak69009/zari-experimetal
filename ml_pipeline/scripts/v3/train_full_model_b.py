import os
import sys
import json
import time
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.v2 as T
from torchvision.models import efficientnet_b2, EfficientNet_B2_Weights
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, confusion_matrix, balanced_accuracy_score

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
DATA_DIR = REPO_ROOT / "ml_pipeline" / "data"
REPORTS_V3_DIR = DATA_DIR / "reports_v3"
CHECKPOINT_DIR = REPO_ROOT / "ml_pipeline" / "checkpoints" / "model_b"
V4_CSV_PATH = DATA_DIR / "dataset_3crop_final_v4_split.csv"

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"✓ Fixed Global Seed: {seed}")

class DiseaseDataset(Dataset):
    def __init__(self, df, label_mapping, transform=None):
        self.df = df.reset_index(drop=True)
        self.label_mapping = label_mapping
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row["image_path"]
        cname = row["class_name"]
        label = self.label_mapping[cname]

        with Image.open(img_path) as img:
            img_rgb = img.convert("RGB")

        if self.transform:
            img_tensor = self.transform(img_rgb)
        else:
            img_tensor = T.functional.to_image(img_rgb)
            img_tensor = T.functional.to_dtype(img_tensor, torch.float32, scale=True)

        return img_tensor, torch.tensor(label, dtype=torch.long)

class EDLEfficientNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = efficientnet_b2(weights=EfficientNet_B2_Weights.DEFAULT)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier[1] = nn.Sequential(
            nn.Dropout(p=0.30),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        logits = self.backbone(x)
        evidence = F.softplus(logits)
        alpha = evidence + 1.0
        S = torch.sum(alpha, dim=1, keepdim=True)
        probs = alpha / S
        uncertainty = logits.shape[1] / S
        return logits, evidence, alpha, S, probs, uncertainty.squeeze(-1)

class EDLLoss(nn.Module):
    def __init__(self, class_weights=None, kl_penalty=0.1):
        super().__init__()
        self.class_weights = class_weights
        self.kl_penalty = kl_penalty

    def forward(self, alpha, target, epoch=1):
        num_classes = alpha.shape[1]
        target_one_hot = F.one_hot(target, num_classes=num_classes).float()
        
        S = torch.sum(alpha, dim=1, keepdim=True)
        log_likelihood = torch.sum(
            target_one_hot * (torch.digamma(S) - torch.digamma(alpha)), dim=1
        )
        
        if self.class_weights is not None:
            weights = self.class_weights[target]
            log_likelihood = log_likelihood * weights

        alpha_tilde = target_one_hot + (1.0 - target_one_hot) * alpha
        S_tilde = torch.sum(alpha_tilde, dim=1, keepdim=True)
        
        kl_div = torch.lgamma(S_tilde) - torch.lgamma(torch.tensor(float(num_classes)).cuda()) \
                 - torch.sum(torch.lgamma(alpha_tilde), dim=1, keepdim=True) \
                 + torch.sum((alpha_tilde - 1.0) * (torch.digamma(alpha_tilde) - torch.digamma(S_tilde)), dim=1, keepdim=True)
        
        anneal = min(1.0, epoch / 10.0) * self.kl_penalty
        loss = torch.mean(log_likelihood + anneal * kl_div.squeeze(-1))
        return loss

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.001, mode="max"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, current_score):
        if self.best_score is None:
            self.best_score = current_score
            return True

        if self.mode == "max":
            improved = (current_score - self.best_score) > self.min_delta
        else:
            improved = (self.best_score - current_score) > self.min_delta

        if improved:
            self.best_score = current_score
            self.counter = 0
            return True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return False

def diagnose_epoch(tr_loss, val_loss, tr_acc, val_acc, tr_f1, val_f1, history):
    acc_gap = tr_acc - val_acc
    f1_gap = tr_f1 - val_f1
    loss_gap = val_loss - tr_loss

    if math.isnan(tr_loss) or math.isnan(val_loss) or math.isinf(tr_loss) or math.isinf(val_loss):
        return "UNSTABLE"

    if tr_acc < 0.50 and val_acc < 0.50:
        return "UNDERFITTING_WARNING"

    if len(history) >= 2:
        prev_val_loss = history[-1]["val_loss"]
        if val_loss > prev_val_loss and tr_loss < history[-1]["train_loss"] and acc_gap > 0.12:
            return "OVERFITTING_WARNING"

    return "HEALTHY"

def train_single_crop_model(crop_name, num_classes, mapping, weights_dict, train_df, val_df, max_epochs=20):
    print(f"\n=====================================================================")
    print(f"  EXECUTING FULL MODEL B TRAINING: {crop_name.upper()} ({num_classes} CLASSES)")
    print(f"=====================================================================")

    # Crop Subsets using GROUND-TRUTH Crop Filter
    tr_crop = train_df[(train_df["crop"] == crop_name) & (train_df["class_name"].isin(mapping.keys()))].copy()
    va_crop = val_df[(val_df["crop"] == crop_name) & (val_df["class_name"].isin(mapping.keys()))].copy()

    print(f"  - Ground-Truth Crop Subset : '{crop_name}'")
    print(f"  - Supervised Train Volume  : {len(tr_crop):,} images")
    print(f"  - Supervised Val Volume    : {len(va_crop):,} images")

    # Transforms
    train_transform = T.Compose([
        T.Resize((256, 256), antialias=True),
        T.RandomHorizontalFlip(p=0.50),
        T.RandomRotation(degrees=15, interpolation=T.InterpolationMode.BILINEAR),
        T.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.90, 1.10)),
        T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10, hue=0.02),
        T.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 1.0)),
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        T.RandomErasing(p=0.10, scale=(0.02, 0.10), value=0)
    ])

    val_transform = T.Compose([
        T.Resize((256, 256), antialias=True),
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    w_tensor = torch.tensor([weights_dict[i] for i in range(num_classes)], dtype=torch.float32).cuda()

    tr_ds = DiseaseDataset(tr_crop, mapping, transform=train_transform)
    va_ds = DiseaseDataset(va_crop, mapping, transform=val_transform)

    tr_loader = DataLoader(tr_ds, batch_size=64, shuffle=True, num_workers=16, pin_memory=True, persistent_workers=True)
    va_loader = DataLoader(va_ds, batch_size=64, shuffle=False, num_workers=16, pin_memory=True, persistent_workers=True)

    model = EDLEfficientNet(num_classes=num_classes).cuda()
    criterion = EDLLoss(class_weights=w_tensor, kl_penalty=0.1)
    scaler = torch.amp.GradScaler('cuda')
    early_stopping = EarlyStopping(patience=5, min_delta=0.001, mode="max")

    best_ckpt_path = CHECKPOINT_DIR / f"best_model_b_{crop_name.lower()}.pth"
    last_ckpt_path = CHECKPOINT_DIR / f"last_model_b_{crop_name.lower()}.pth"
    hist_csv_path = REPORTS_V3_DIR / f"model_b_{crop_name.lower()}_training_history.csv"
    val_report_md_path = REPORTS_V3_DIR / f"model_b_{crop_name.lower()}_validation_report.md"

    history = []
    best_val_f1 = 0.0
    best_epoch = 0
    STAGE_1_EPOCHS = 3

    for epoch in range(1, max_epochs + 1):
        epoch_start = time.time()
        stage = "STAGE_1_HEAD_WARMUP" if epoch <= STAGE_1_EPOCHS else "STAGE_2_FINE_TUNING"

        if epoch == 1:
            print(f"\n--- STAGE 1: Freezing Backbone for {crop_name} (Epochs 1..{STAGE_1_EPOCHS}) ---")
            for name, param in model.named_parameters():
                if "classifier" not in name:
                    param.requires_grad = False
            optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3, weight_decay=1e-4)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2, min_lr=1e-7)

        elif epoch == STAGE_1_EPOCHS + 1:
            print(f"\n--- STAGE 2: Unfreezing Backbone for {crop_name} (Epochs {STAGE_1_EPOCHS+1}..{max_epochs}) ---")
            for param in model.parameters():
                param.requires_grad = True

            backbone_params = [p for n, p in model.named_parameters() if "classifier" not in n]
            head_params = [p for n, p in model.named_parameters() if "classifier" in n]

            optimizer = optim.AdamW([
                {"params": backbone_params, "lr": 1e-4},
                {"params": head_params, "lr": 1e-3}
            ], weight_decay=1e-4)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2, min_lr=1e-7)

        # Training Pass
        model.train()
        tr_loss = 0.0
        tr_preds, tr_targets = [], []
        tr_uncs = []

        for x, y in tr_loader:
            x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                logits, evidence, alpha, S, probs, uncertainty = model(x)
                loss = criterion(alpha, y, epoch=epoch)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            tr_loss += loss.item()
            preds = torch.argmax(probs, dim=1)
            tr_preds.extend(preds.cpu().numpy())
            tr_targets.extend(y.cpu().numpy())
            tr_uncs.extend(uncertainty.detach().cpu().numpy())

        tr_loss /= len(tr_loader)
        tr_acc = accuracy_score(tr_targets, tr_preds)
        tr_f1 = f1_score(tr_targets, tr_preds, average="macro")

        # Validation Pass
        model.eval()
        val_loss = 0.0
        val_preds, val_targets = [], []
        val_uncs = []

        with torch.no_grad():
            for x, y in va_loader:
                x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
                with torch.amp.autocast('cuda'):
                    logits, evidence, alpha, S, probs, uncertainty = model(x)
                    loss = criterion(alpha, y, epoch=epoch)

                val_loss += loss.item()
                preds = torch.argmax(probs, dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_targets.extend(y.cpu().numpy())
                val_uncs.extend(uncertainty.detach().cpu().numpy())

        val_loss /= len(va_loader)
        val_acc = accuracy_score(val_targets, val_preds)
        val_f1 = f1_score(val_targets, val_preds, average="macro")

        scheduler.step(val_loss)
        lr_bb = optimizer.param_groups[0]["lr"] if len(optimizer.param_groups) > 1 else 0.0
        lr_hd = optimizer.param_groups[-1]["lr"]

        diag_state = diagnose_epoch(tr_loss, val_loss, tr_acc, val_acc, tr_f1, val_f1, history)
        epoch_dur = time.time() - epoch_start

        history.append({
            "epoch": epoch,
            "stage": stage,
            "train_loss": round(tr_loss, 4),
            "val_loss": round(val_loss, 4),
            "train_accuracy": round(tr_acc, 4),
            "val_accuracy": round(val_acc, 4),
            "train_macro_f1": round(tr_f1, 4),
            "val_macro_f1": round(val_f1, 4),
            "mean_val_uncertainty": round(float(np.mean(val_uncs)), 4),
            "learning_rate_backbone": lr_bb,
            "learning_rate_head": lr_hd,
            "generalization_gap": round(tr_f1 - val_f1, 4),
            "diagnostic_state": diag_state
        })

        print(f"[{crop_name}] Epoch {epoch:02d}/{max_epochs} [{stage[:7]}] | Tr Loss: {tr_loss:.4f} Acc: {tr_acc*100:.2f}% F1: {tr_f1:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc*100:.2f}% F1: {val_f1:.4f} | Diag: {diag_state} ({epoch_dur:.1f}s)")

        ckpt_payload = {
            "epoch": epoch,
            "stage": stage,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "val_macro_f1": val_f1,
            "val_accuracy": val_acc,
            "class_mapping": mapping,
            "crop": crop_name,
            "config": {"batch_size": 64, "seed": 42, "manifest": "dataset_3crop_final_v4_split.csv"}
        }
        torch.save(ckpt_payload, last_ckpt_path)

        is_best = early_stopping(val_f1)
        if is_best:
            best_val_f1 = val_f1
            best_epoch = epoch
            ckpt_payload["best_val_macro_f1"] = best_val_f1
            torch.save(ckpt_payload, best_ckpt_path)
            print(f"  ★ NEW BEST {crop_name.upper()} CHECKPOINT SAVED at Epoch {epoch} (Val Macro F1: {best_val_f1:.4f})")

        if early_stopping.early_stop:
            print(f"\n✋ Early Stopping Triggered for {crop_name} at Epoch {epoch}! Validation Macro F1 did not improve for {early_stopping.patience} epochs.")
            history[-1]["diagnostic_state"] = "EARLY_STOPPING"
            break

    # Save Per-Crop CSV History
    hist_df = pd.DataFrame(history)
    hist_df.to_csv(hist_csv_path, index=False)
    print(f"✓ Saved {crop_name} Training History CSV: {hist_csv_path.relative_to(REPO_ROOT)}")

    # Final Validation Evaluation on BEST Checkpoint ONLY
    print(f"\nReloading BEST {crop_name} Checkpoint for Final Validation Evaluation...")
    best_ckpt = torch.load(best_ckpt_path)
    model.load_state_dict(best_ckpt["model_state_dict"])
    model.eval()

    eval_preds, eval_targets, eval_uncs = [], [], []
    with torch.no_grad():
        for x, y in va_loader:
            x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
            with torch.amp.autocast('cuda'):
                logits, evidence, alpha, S, probs, uncertainty = model(x)
            preds = torch.argmax(probs, dim=1)
            eval_preds.extend(preds.cpu().numpy())
            eval_targets.extend(y.cpu().numpy())
            eval_uncs.extend(uncertainty.cpu().numpy())

    final_val_acc = accuracy_score(eval_targets, eval_preds)
    bal_acc = balanced_accuracy_score(eval_targets, eval_preds)
    class_names = [k for k, v in sorted(mapping.items(), key=lambda item: item[1])]
    precision_p, recall_r, f1_f, support_s = precision_recall_fscore_support(eval_targets, eval_preds, average=None, labels=list(range(num_classes)))
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(eval_targets, eval_preds, average="macro")
    cm = confusion_matrix(eval_targets, eval_preds)

    mean_unc_correct = np.mean(np.array(eval_uncs)[np.array(eval_preds) == np.array(eval_targets)])
    mean_unc_incorrect = np.mean(np.array(eval_uncs)[np.array(eval_preds) != np.array(eval_targets)]) if sum(np.array(eval_preds) != np.array(eval_targets)) > 0 else 0.0

    print(f"=====================================================================")
    print(f"  MODEL B {crop_name.upper()} FINAL VALIDATION RESULTS (BEST EPOCH {best_epoch})")
    print(f"=====================================================================")
    print(f"  Overall Validation Accuracy : {final_val_acc*100:.2f}%")
    print(f"  Balanced Accuracy           : {bal_acc*100:.2f}%")
    print(f"  Macro Precision             : {macro_p:.4f}")
    print(f"  Macro Recall                : {macro_r:.4f}")
    print(f"  Macro F1-Score              : {macro_f1:.4f}")
    print(f"  Mean Correct Uncertainty   : {mean_unc_correct:.4f}")
    print(f"  Mean Incorrect Uncertainty : {mean_unc_incorrect:.4f}")
    print(f"=====================================================================\n")

    # Generate Crop Validation Report Markdown
    lines = []
    lines.append(f"# ZARI.ai — Model B {crop_name} Final Validation Report\n")
    lines.append(f"**Training Date**: August 17, 2026  ")
    lines.append(f"**Crop Target**: **Model B {crop_name}** ({num_classes} Supervised Classes)  ")
    lines.append(f"**Best Epoch**: **Epoch {best_epoch}** (Best Val Macro F1: **{macro_f1:.4f}**)  ")
    lines.append(f"**Validation Accuracy**: **{final_val_acc*100:.2f}%** | **Balanced Acc**: **{bal_acc*100:.2f}%**  \n")
    lines.append("---\n")
    lines.append("## 1. Per-Class Validation Breakdown\n")
    lines.append("| Disease Class Label | Label ID | Val Support | Precision | Recall | F1-Score |")
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: |")
    for i, cname in enumerate(class_names):
        lines.append(f"| **{cname}** | {i} | {support_s[i]:,} | {precision_p[i]:.4f} | {recall_r[i]:.4f} | **{f1_f[i]:.4f}** |")
    lines.append("\n---\n")
    lines.append("## 2. Validation Confusion Matrix\n")
    lines.append("```text\n" + str(cm) + "\n```\n")
    lines.append("---\n")
    lines.append("## 3. EDL Epistemic Uncertainty Profile\n")
    lines.append(f"- **Mean Uncertainty (Correct Predictions)**: `{mean_unc_correct:.4f}`")
    lines.append(f"- **Mean Uncertainty (Incorrect Predictions)**: `{mean_unc_incorrect:.4f}`\n")
    lines.append("---\n")
    lines.append("## 4. Per-Epoch History Table\n")
    lines.append(hist_df.to_markdown(index=False))

    val_report_md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"✓ Saved {crop_name} Validation Report: {val_report_md_path.relative_to(REPO_ROOT)}")

    return {
        "crop": crop_name,
        "best_epoch": best_epoch,
        "best_val_macro_f1": macro_f1,
        "val_accuracy": final_val_acc,
        "balanced_accuracy": bal_acc,
        "overfitting_state": hist_df.iloc[-1]["diagnostic_state"]
    }

def run_full_model_b_training():
    print("=====================================================================")
    print("  ZARI.ai — FULL MODEL B DISEASE CLASSIFIER TRAINING EXECUTION")
    print("=====================================================================\n")

    set_seed(42)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_V3_DIR.mkdir(parents=True, exist_ok=True)

    df_full = pd.read_csv(V4_CSV_PATH, low_memory=False)
    train_df = df_full[df_full["split"] == "train"].copy()
    val_df = df_full[df_full["split"] == "val"].copy()

    # Mappings & Weights
    tomato_mapping = {
        "Tomato_Bacterial_Spot": 0, "Tomato_Early_Blight": 1, "Tomato_Fusarium_Wilt": 2,
        "Tomato_Healthy": 3, "Tomato_Late_Blight": 4, "Tomato_Leaf_Mold": 5,
        "Tomato_Miner": 6, "Tomato_Mosaic_Virus": 7, "Tomato_Septoria_Leaf_Spot": 8,
        "Tomato_Spider_Mites": 9, "Tomato_Target_Spot": 10, "Tomato_Verticillium_Wilt": 11,
        "Tomato_Yellow_Leaf_Curl_Virus": 12
    }
    potato_supported_mapping = {
        "Potato_Early_Blight": 0, "Potato_Late_Blight": 1, "Potato_Healthy": 2
    }
    pepper_mapping = {
        "Pepper_Bacterial_Spot": 0, "Pepper_Cercospora_Leaf_Spot": 1, "Pepper_Healthy": 2,
        "Pepper_Leaf_Curl": 3, "Pepper_Nutrition_Deficiency": 4, "Pepper_Powdery_Mildew": 5
    }

    tomato_weights_dict = {0: 0.6833, 1: 0.8704, 2: 6.6286, 3: 0.7586, 4: 0.7848, 5: 0.8711, 6: 1.4201, 7: 6.3176, 8: 0.7512, 9: 1.1645, 10: 1.9184, 11: 5.1911, 12: 0.3663}
    potato_weights_dict = {0: 0.7835, 1: 0.8481, 2: 1.8365}
    pepper_weights_dict = {0: 0.3326, 1: 0.8932, 2: 0.9064, 3: 3.2525, 4: 3.1063, 5: 7.0887}

    # Train All 3 Crop Models Sequentially
    results = []

    res_tom = train_single_crop_model("Tomato", 13, tomato_mapping, tomato_weights_dict, train_df, val_df, max_epochs=20)
    results.append(res_tom)

    res_pot = train_single_crop_model("Potato", 3, potato_supported_mapping, potato_weights_dict, train_df, val_df, max_epochs=20)
    results.append(res_pot)

    res_pep = train_single_crop_model("Pepper", 6, pepper_mapping, pepper_weights_dict, train_df, val_df, max_epochs=20)
    results.append(res_pep)

    # Post-Training Audit
    print("\n=====================================================================")
    print("  MODEL B FINAL SAFETY AUDIT VERIFICATION")
    print("=====================================================================")
    audit_passed = True
    for r in results:
        ckpt_p = CHECKPOINT_DIR / f"best_model_b_{r['crop'].lower()}.pth"
        if not ckpt_p.exists():
            print(f"  ❌ Missing Checkpoint: {ckpt_p}")
            audit_passed = False
        else:
            print(f"  ✓ Checkpoint Exists & Verified: {ckpt_p.name} (Best Epoch {r['best_epoch']}, Val F1: {r['best_val_macro_f1']:.4f})")

    final_pass_status = "MODEL_B_FULL_TRAINING_PASS" if audit_passed else "MODEL_B_FULL_TRAINING_FAILED"

    print(f"\n============================================================")
    print(f"{final_pass_status}")
    print("============================================================")
    for r in results:
        print(f"\n{r['crop']}:")
        print(f"  best epoch       : {r['best_epoch']}")
        print(f"  best val Macro F1: {r['best_val_macro_f1']:.4f}")
        print(f"  val accuracy     : {r['val_accuracy']*100:.2f}%")
        print(f"  overfitting state: {r['overfitting_state']}")
    print("============================================================")

if __name__ == "__main__":
    run_full_model_b_training()
