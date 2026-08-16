"""Phase 2 Improved Domain Adaptation & Evidential Deep Learning (EDL) Training for ZARI.ai.
---------------------------------------------------------------------------------------
Key Enhancements in Phase 2 Improved:
1. Partial Backbone Unfreeze:
   - Freeze early blocks (0-4) to retain general edge/texture representations.
   - Unfreeze last blocks ("blocks.5", "blocks.6", "conv_head") to learn high-level
     wheat-specific disease features.
2. Two Learning Rate Parameter Groups:
   - Backbone unfreezed params: lr = 1e-5 (fine-tune without destroying pretrained representations)
   - EDL Classifier head: lr = 3e-4 (more stable convergence, preventing overfitting)
3. Automated Early Stopping:
   - Stop training when validation accuracy fails to improve for patience=2 consecutive epochs.
4. Comprehensive Test Set Evaluation:
   - Evaluate best checkpoint on 6,648 field test set images and produce comparative report
     at ml_pipeline/ANALYSIS_COMPLETE/reports/phase2_improved_comparison.txt.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

try:
    import timm
except ImportError:
    timm = None

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"
MODELS_DIR = SCRIPT_DIR / "models"
LOGS_DIR = SCRIPT_DIR / "logs"
REPORTS_DIR = SCRIPT_DIR / "ANALYSIS_COMPLETE" / "reports"

INPUT_CSV = DATA_DIR / "dataset_final_training.csv"
WEIGHTS_JSON = DATA_DIR / "class_weights.json"
CLASS_MAP_JSON = DATA_DIR / "class_map_final.json"
RAW_ROOT = DATA_DIR / "raw"

PHASE1_BACKBONE_PATH = MODELS_DIR / "phase1_backbone.pth"
IMPROVED_MODEL_PATH = MODELS_DIR / "phase2_improved_model.pth"
IMPROVED_HISTORY_PATH = LOGS_DIR / "phase2_improved_history.json"
COMPARISON_REPORT_PATH = REPORTS_DIR / "phase2_improved_comparison.txt"

NUM_CLASSES = 67  # 67 Head Field Classes


def resolve_image_path(image_path_str: str) -> Path:
    """Resolve image path handling absolute, relative, and legacy dataset root paths."""
    candidate = Path(image_path_str)
    if candidate.exists():
        return candidate

    match = re.search(r"raw[\\/](.+)$", str(image_path_str), flags=re.IGNORECASE)
    if match:
        suffix = match.group(1).replace("\\", "/")
        candidate = RAW_ROOT / Path(suffix)
        if candidate.exists():
            return candidate

    raise FileNotFoundError(f"Image path cannot be resolved: {image_path_str}")


class FieldDataset(Dataset):
    """Custom PyTorch Dataset for ZARI.ai Phase 2 Field Fine-Tuning."""

    def __init__(self, df: pd.DataFrame, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.image_paths = self.df["image_path"].astype(str).tolist()
        self.labels = self.df["class_id"].astype(int).tolist()

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        image_path_str = self.image_paths[idx]
        label = self.labels[idx]

        resolved_path = resolve_image_path(image_path_str)
        image = Image.open(resolved_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


def edl_kl_divergence(alpha: torch.Tensor, target_one_hot: torch.Tensor) -> torch.Tensor:
    """Compute KL Divergence regularizer for Dirichlet parameters against uniform prior."""
    device = alpha.device
    num_classes = alpha.size(1)

    alpha_tilde = target_one_hot + (1.0 - target_one_hot) * alpha
    sum_alpha_tilde = torch.sum(alpha_tilde, dim=1, keepdim=True)

    kl = (
        torch.lgamma(sum_alpha_tilde)
        - torch.lgamma(torch.tensor(float(num_classes), device=device))
        - torch.sum(torch.lgamma(alpha_tilde), dim=1, keepdim=True)
        + torch.sum(
            (alpha_tilde - 1.0) * (torch.digamma(alpha_tilde) - torch.digamma(sum_alpha_tilde)),
            dim=1,
            keepdim=True,
        )
    )
    return kl.squeeze(-1)


def edl_loss_fn(
    logits: torch.Tensor,
    target_labels: torch.Tensor,
    epoch: int,
    max_epochs: int,
    weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute Evidential Deep Learning (EDL) Loss with KL annealing.

    Returns:
        (total_loss, mean_uncertainty)
    """
    num_classes = logits.size(1)
    evidence = F.softplus(logits)
    alpha = evidence + 1.0
    S = torch.sum(alpha, dim=1, keepdim=True)

    uncertainty = float(num_classes) / S.squeeze(-1)
    mean_uncertainty = torch.mean(uncertainty)

    target_one_hot = F.one_hot(target_labels, num_classes=num_classes).float()

    ace_loss = torch.sum(target_one_hot * (torch.digamma(S) - torch.digamma(alpha)), dim=1)
    annealing_coef = min(1.0, float(epoch + 1) / 10.0)
    kl_loss = edl_kl_divergence(alpha, target_one_hot)

    loss_per_sample = ace_loss + annealing_coef * kl_loss

    if weights is not None:
        sample_weights = weights[target_labels]
        loss_per_sample = loss_per_sample * sample_weights

    return torch.mean(loss_per_sample), mean_uncertainty


def main() -> None:
    print("=" * 75)
    print("  ZARI.ai — PHASE 2 IMPROVED TRAINING (PARTIAL UNFREEZE & LOWER LR)")
    print("=" * 75)

    if timm is None:
        raise ImportError("timm module is required. Please install timm: pip install timm")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Missing input dataset CSV at {INPUT_CSV}")
    if not WEIGHTS_JSON.exists():
        raise FileNotFoundError(f"Missing class weights JSON at {WEIGHTS_JSON}")
    if not PHASE1_BACKBONE_PATH.exists():
        raise FileNotFoundError(f"Missing Phase 1 backbone weights at {PHASE1_BACKBONE_PATH}")

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing compute device: {device}")
    if device.type == "cuda":
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"VRAM Available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # 1. Load Dataset
    df = pd.read_csv(INPUT_CSV)
    print(f"\n[STEP 1] Loaded dataset: {INPUT_CSV} ({len(df):,} total rows)")

    train_df = df[(df["split"] == "train") & (df["class_id"] >= 0)].copy()
    val_df = df[(df["split"] == "val") & (df["class_id"] >= 0)].copy()
    test_df = df[(df["split"] == "test") & (df["class_id"] >= 0)].copy()

    print(f"✓ Field Training samples  : {len(train_df):,}")
    print(f"✓ Field Validation samples: {len(val_df):,}")
    print(f"✓ Field Test samples      : {len(test_df):,}")

    # 2. Data Transforms (Same as Phase 2 Baseline)
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(384, scale=(0.75, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
        transforms.RandomRotation(degrees=15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = FieldDataset(train_df, transform=train_transform)
    val_dataset = FieldDataset(val_df, transform=val_transform)

    batch_size = 64 if device.type == "cuda" else 16
    num_workers = min(8, os.cpu_count() or 4)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=(device.type == "cuda")
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=(device.type == "cuda")
    )

    print(f"✓ Batch Size: {batch_size} (Num Workers: {num_workers})")

    # 3. Model Architecture & Partial Backbone Unfreezing
    print("\n[STEP 2] Initializing Backbone & Performing Partial Unfreeze...")
    backbone = timm.create_model("tf_efficientnetv2_s.in21k_ft_in1k", pretrained=False)
    backbone.reset_classifier(0)

    # Load Phase 1 pretrained backbone weights
    backbone_state = torch.load(PHASE1_BACKBONE_PATH, map_location="cpu")
    backbone.load_state_dict(backbone_state, strict=False)
    print(f"✓ Successfully loaded Phase 1 backbone weights from {PHASE1_BACKBONE_PATH.name}")

    # Freeze everything first
    for param in backbone.parameters():
        param.requires_grad = False

    # Unfreeze last 2 blocks + conv_head
    unfreeze_patterns = ["blocks.5", "blocks.6", "conv_head"]

    for name, param in backbone.named_parameters():
        if any(pattern in name for pattern in unfreeze_patterns):
            param.requires_grad = True

    # Count trainable params in backbone
    trainable_backbone = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
    print(f"Backbone trainable: {trainable_backbone:,}")

    # Attach EDL Linear Head
    edl_head = nn.Linear(1280, NUM_CLASSES)

    # Move to device
    backbone = backbone.to(device)
    edl_head = edl_head.to(device)

    # Combine into model wrapper for forward pass
    class ZariEDLModel(nn.Module):
        def __init__(self, bb: nn.Module, head: nn.Module):
            super().__init__()
            self.backbone = bb
            self.head = head

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            feats = self.backbone(x)
            return self.head(feats)

    model = ZariEDLModel(backbone, edl_head)

    # 4. Class Weights
    with open(WEIGHTS_JSON, "r", encoding="utf-8") as f:
        weights_data = json.load(f)

    head_weights = [weights_data["head_weights"][str(i)] for i in range(NUM_CLASSES)]
    weights_tensor = torch.tensor(head_weights, dtype=torch.float32).to(device)

    # 5. Optimizer with Two LR Groups
    optimizer = AdamW([
        {
            "params": [p for p in backbone.parameters() if p.requires_grad],
            "lr": 1e-5,  # Very low — fine-tune, don't destroy
            "weight_decay": 0.01,
        },
        {
            "params": edl_head.parameters(),
            "lr": 3e-4,  # Lower than before (was 1e-3)
            "weight_decay": 0.01,
        }
    ])

    use_amp = (device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    grad_accum_steps = 2
    max_epochs = 15
    patience = 2

    training_history: dict[str, list] = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": [],
        "mean_uncertainty": [],
        "lr_backbone": [],
        "lr_head": [],
        "time_seconds": [],
    }

    # 6. Training Loop with Early Stopping
    print(f"\n[STEP 3] Starting Training (Max Epochs: {max_epochs}, Early Stopping Patience: {patience})...")
    best_val_acc = 0.0
    no_improvement = 0
    start_time = time.time()

    for epoch in range(max_epochs):
        epoch_start = time.time()

        # --- TRAINING PHASE ---
        model.train()
        running_loss = 0.0
        running_u = 0.0
        optimizer.zero_grad()

        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(images)
                loss, batch_u = edl_loss_fn(logits, labels, epoch, max_epochs, weights=weights_tensor)
                loss = loss / grad_accum_steps

            scaler.scale(loss).backward()

            if (batch_idx + 1) % grad_accum_steps == 0 or (batch_idx + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            running_loss += loss.item() * grad_accum_steps
            running_u += batch_u.item()

        train_loss = running_loss / len(train_loader)
        train_u = running_u / len(train_loader)

        # --- VALIDATION PHASE ---
        model.eval()
        correct = 0
        total = 0
        val_loss = 0.0
        val_u = 0.0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                with torch.amp.autocast("cuda", enabled=use_amp):
                    logits = model(images)
                    loss, batch_u = edl_loss_fn(logits, labels, epoch, max_epochs, weights=None)

                val_loss += loss.item()
                val_u += batch_u.item()

                evidence = F.softplus(logits)
                alpha = evidence + 1.0
                preds = alpha.argmax(dim=1)
                correct += int((preds == labels).sum().item())
                total += int(labels.size(0))

        val_acc = correct / total if total > 0 else 0.0
        val_avg_loss = val_loss / len(val_loader)
        val_avg_u = val_u / len(val_loader)
        epoch_dur = time.time() - epoch_start

        lr_bb = float(optimizer.param_groups[0]["lr"])
        lr_hd = float(optimizer.param_groups[1]["lr"])

        training_history["epoch"].append(epoch + 1)
        training_history["train_loss"].append(round(float(train_loss), 4))
        training_history["val_loss"].append(round(float(val_avg_loss), 4))
        training_history["val_accuracy"].append(round(float(val_acc), 4))
        training_history["mean_uncertainty"].append(round(float(val_avg_u), 4))
        training_history["lr_backbone"].append(round(float(lr_bb), 6))
        training_history["lr_head"].append(round(float(lr_hd), 6))
        training_history["time_seconds"].append(round(float(epoch_dur), 2))

        # Best Checkpoint Tracking
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            no_improvement = 0
            torch.save({
                'backbone': backbone.state_dict(),
                'edl_head': edl_head.state_dict(),
            }, IMPROVED_MODEL_PATH)
            best_tag = "⭐ BEST"
        else:
            no_improvement += 1
            best_tag = f"(no imp: {no_improvement}/{patience})"

        status_str = (
            f"Epoch {epoch+1:02d}/{max_epochs:02d} | Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_avg_loss:.4f} | Val Acc: {val_acc*100:.2f}% | "
            f"Uncertainty: {val_avg_u:.4f} | Time: {epoch_dur:.1f}s {best_tag}"
        )
        print("-" * 75)
        print(status_str)
        print("-" * 75)

        # Early stopping check
        if no_improvement >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    total_training_time = time.time() - start_time
    print(f"\n✓ Training completed in {total_training_time/60:.2f} minutes.")
    print(f"✓ Best Validation Accuracy: {best_val_acc*100:.2f}%")

    # Save History JSON
    IMPROVED_HISTORY_PATH.write_text(json.dumps(training_history, indent=2), encoding="utf-8")
    print(f"✓ Saved training history to: {IMPROVED_HISTORY_PATH}")

    # 7. Evaluate Best Checkpoint on Test Set (6,648 Field Images)
    print("\n[STEP 4] Evaluating Best Checkpoint on Test Set (6,648 Images)...")

    # Load Class Map
    class_id_to_name: dict[int, str] = {}
    if CLASS_MAP_JSON.exists():
        with open(CLASS_MAP_JSON, "r", encoding="utf-8") as f:
            class_map_data = json.load(f)
            head_classes = class_map_data.get("head_classes", {})
            for name, cid in head_classes.items():
                if isinstance(cid, int) and cid >= 0:
                    class_id_to_name[cid] = name

    test_dataset = FieldDataset(test_df, transform=val_transform)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=(device.type == "cuda")
    )

    # Load best checkpoint
    checkpoint = torch.load(IMPROVED_MODEL_PATH, map_location=device)
    backbone.load_state_dict(checkpoint['backbone'])
    edl_head.load_state_dict(checkpoint['edl_head'])
    model.eval()

    all_y_true: list[int] = []
    all_y_pred: list[int] = []
    all_uncertainties: list[float] = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(images)
                evidence = F.softplus(logits)
                alpha = evidence + 1.0
                S = torch.sum(alpha, dim=1, keepdim=True)
                probs = alpha / S
                uncertainty = float(NUM_CLASSES) / S.squeeze(-1)
                preds = probs.argmax(dim=1)

            all_y_true.extend(labels.numpy().tolist())
            all_y_pred.extend(preds.cpu().numpy().tolist())
            all_uncertainties.extend(uncertainty.cpu().numpy().tolist())

    y_true = np.array(all_y_true, dtype=int)
    y_pred = np.array(all_y_pred, dtype=int)
    u_vals = np.array(all_uncertainties, dtype=float)
    is_correct = (y_true == y_pred).astype(int)

    test_acc = float(accuracy_score(y_true, y_pred))
    test_f1 = float(f1_score(y_true, y_pred, average="macro"))
    test_auroc = float(roc_auc_score(1 - is_correct, u_vals))

    # Baseline comparison metrics
    baseline_metrics = {
        "Overall": 97.32,
        "Wheat_Black_Rust": 77.14,
        "Wheat_Brown_Rust": 87.05,
        "Wheat_Tan_Spot": 66.27,
        "Wheat_Leaf_Blight": 69.32,
        "Wheat_Septoria": 89.08,
        "AUROC": 0.9527,
    }

    target_wheat_classes = [
        "Wheat_Black_Rust",
        "Wheat_Brown_Rust",
        "Wheat_Tan_Spot",
        "Wheat_Leaf_Blight",
        "Wheat_Septoria",
    ]

    improved_wheat_accs: dict[str, float] = {}
    for cname in target_wheat_classes:
        cid = [k for k, v in class_id_to_name.items() if v == cname][0]
        mask = (y_true == cid)
        count = int(mask.sum())
        if count > 0:
            c_acc = float((y_pred[mask] == cid).sum()) / count * 100.0
        else:
            c_acc = 0.0
        improved_wheat_accs[cname] = c_acc

    improved_metrics = {
        "Overall": test_acc * 100.0,
        "Wheat_Black_Rust": improved_wheat_accs["Wheat_Black_Rust"],
        "Wheat_Brown_Rust": improved_wheat_accs["Wheat_Brown_Rust"],
        "Wheat_Tan_Spot": improved_wheat_accs["Wheat_Tan_Spot"],
        "Wheat_Leaf_Blight": improved_wheat_accs["Wheat_Leaf_Blight"],
        "Wheat_Septoria": improved_wheat_accs["Wheat_Septoria"],
        "AUROC": test_auroc,
    }

    # 8. Format & Save Comparison Report
    report_lines = [
        "================================================================================",
        "ZARI.ai — PHASE 2 IMPROVED MODEL TEST EVALUATION & COMPARISON REPORT",
        "================================================================================",
        f"Date / Timestamp           : {datetime.now().isoformat()}",
        f"Evaluation Dataset         : Test Split ({len(y_true):,} Field Images)",
        f"Evaluated Model Checkpoint : {IMPROVED_MODEL_PATH.name}",
        f"Training Setup             : Partial Unfreeze (blocks.5, blocks.6, conv_head)",
        "                             Two LR Groups (Backbone: 1e-5, Head: 3e-4)",
        "                             Early Stopping (patience=2)",
        "",
        "METRICS COMPARISON TABLE:",
        "--------------------------------------------------------------------------------",
        f"{'Metric':<25} | {'Current':<12} | {'Improved':<12} | {'Change':<12}",
        "-" * 70,
    ]

    for key in baseline_metrics:
        curr_val = baseline_metrics[key]
        impr_val = improved_metrics[key]

        if key == "AUROC":
            chg = impr_val - curr_val
            curr_str = f"{curr_val:.4f}"
            impr_str = f"{impr_val:.4f}"
            chg_str = f"{chg:+0.4f}"
        else:
            chg = impr_val - curr_val
            curr_str = f"{curr_val:6.2f}%"
            impr_str = f"{impr_val:6.2f}%"
            chg_str = f"{chg:+6.2f}%"

        report_lines.append(f"{key:<25} | {curr_str:<12} | {impr_str:<12} | {chg_str:<12}")

    # Verdict Logic
    overall_change = improved_metrics["Overall"] - baseline_metrics["Overall"]
    wheat_improvements = [
        key for key in target_wheat_classes if improved_metrics[key] > baseline_metrics[key]
    ]

    is_unfreeze_beneficial = (overall_change >= 0.0 or len(wheat_improvements) >= 2)

    verdict_lines = [
        "",
        "================================================================================",
        "FINAL VERDICT & RECOMMENDATION",
        "================================================================================",
        f"1. Wheat Confusion Pairs Status  : {'IMPROVED' if len(wheat_improvements) > 0 else 'STABLE'}",
        f"   - Wheat classes with accuracy gains: {', '.join(wheat_improvements) if wheat_improvements else 'None'}.",
        f"2. Partial Unfreeze & Lower LR   : {'HELPFUL' if is_unfreeze_beneficial else 'NEUTRAL'}",
        f"   - Overall Accuracy: {test_acc*100:.2f}% ({overall_change:+0.2f}% vs Baseline 97.32%).",
        f"3. Model Deployment Recommendation: {'KEEP THIS IMPROVED MODEL' if is_unfreeze_beneficial else 'RETAIN BASELINE EDL MODEL'}",
        "================================================================================",
    ]

    report_lines.extend(verdict_lines)
    COMPARISON_REPORT_PATH.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"✓ Saved comparison report to: {COMPARISON_REPORT_PATH}")

    print("\n" + "=" * 75)
    print("  PHASE 2 IMPROVED COMPARISON TABLE")
    print("=" * 75)
    for line in report_lines[12:21]:
        print(line)
    print("\n" + verdict_lines[4])
    print(f"✓ Model Checkpoint : {IMPROVED_MODEL_PATH}")
    print(f"✓ History JSON     : {IMPROVED_HISTORY_PATH}")
    print(f"✓ Comparison Report: {COMPARISON_REPORT_PATH}")
    print("\n✅ PHASE 2 IMPROVED PIPELINE EXECUTION COMPLETE!")


if __name__ == "__main__":
    main()
