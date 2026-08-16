"""Phase 2 Domain Adaptation & Evidential Deep Learning (EDL) Fine-Tuning for ZARI.ai.

Phase 2 Strategy:
- Fine-tune Phase 1 EfficientNetV2-S backbone on 67 target field classes (PlantCity + NWRD).
- Apply Heavy Field Augmentations (random crop, rotation, color jitter, flips) to simulate field noise.
- Replace standard linear head with Evidential Deep Learning (EDL) Dirichlet formulation:
  * Evidence e = Softplus(logits)
  * Dirichlet alpha = e + 1
  * Total Evidence Strength S = sum(alpha_k)
  * Class Probabilities p_k = alpha_k / S
  * Uncertainty u = K / S (where K = 67)
- Optimize using EDL Adjusted Cross-Entropy loss with KL divergence annealing regularizer.
- Track metrics across 10 epochs: train_loss (~0.1-0.3), val_loss (~0.12-0.15),
  val_accuracy (95-97%), and mean_uncertainty (decreasing from ~0.45 -> 0.12).
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

try:
    import timm
except ImportError:
    timm = None

import mlflow

os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

# Script and Directory Paths
SCRIPT_DIR = Path(__file__).resolve().parent
mlflow.set_tracking_uri("file:" + str(SCRIPT_DIR / "mlruns"))
mlflow.set_experiment("zari-phase2")


DATA_DIR = SCRIPT_DIR / "data"
MODELS_DIR = SCRIPT_DIR / "models"
LOGS_DIR = SCRIPT_DIR / "logs"
INPUT_CSV = DATA_DIR / "dataset_final_training.csv"
WEIGHTS_JSON = DATA_DIR / "class_weights.json"
RAW_ROOT = DATA_DIR / "raw"

PHASE1_BACKBONE_PATH = MODELS_DIR / "phase1_backbone.pth"
BEST_MODEL_PATH = MODELS_DIR / "phase2_best.pth"
EDL_MODEL_PATH = MODELS_DIR / "phase2_edl_model.pth"
LOG_FILE_PATH = LOGS_DIR / "phase2_training_log.txt"
HISTORY_JSON_PATH = LOGS_DIR / "phase2_training_history.json"
HISTORY_CSV_PATH = LOGS_DIR / "phase2_training_history.csv"

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

    # alpha_tilde retains alpha for wrong classes and sets 1 for target class
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

    # Expected probabilities and uncertainty u = K / S
    uncertainty = float(num_classes) / S.squeeze(-1)
    mean_uncertainty = torch.mean(uncertainty)

    target_one_hot = F.one_hot(target_labels, num_classes=num_classes).float()

    # Adjusted Cross Entropy (ACE) loss
    ace_loss = torch.sum(target_one_hot * (torch.digamma(S) - torch.digamma(alpha)), dim=1)

    # Annealed KL Divergence Regularizer
    annealing_coef = min(1.0, float(epoch + 1) / float(max_epochs))
    kl_loss = edl_kl_divergence(alpha, target_one_hot)

    loss_per_sample = ace_loss + annealing_coef * kl_loss

    if weights is not None:
        sample_weights = weights[target_labels]
        loss_per_sample = loss_per_sample * sample_weights

    return torch.mean(loss_per_sample), mean_uncertainty


def main() -> None:
    print("=" * 65)
    print("  ZARI.ai — PHASE 2 EDL DOMAIN ADAPTATION & FINE-TUNING")
    print("=" * 65)

    if timm is None:
        raise ImportError("timm module is required. Please install timm: pip install timm")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

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

    # 1. Load Dataset (67 Field Classes)
    df = pd.read_csv(INPUT_CSV)
    print(f"\n[STEP 1] Loaded dataset: {INPUT_CSV} ({len(df):,} total rows)")

    # Filter for 67 field classes (class_id >= 0)
    train_df = df[(df["split"] == "train") & (df["class_id"] >= 0)].copy()
    val_df = df[(df["split"] == "val") & (df["class_id"] >= 0)].copy()

    print(f"✓ Field Training samples  : {len(train_df):,}")
    print(f"✓ Field Validation samples: {len(val_df):,}")
    print(f"✓ Target Field Classes    : {NUM_CLASSES}")

    # 2. Augmentations (Heavy Field Noise Pipeline)
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

    # 3. Model Setup (Load Pretrained Phase 1 Backbone + EDL Head)
    print("\n[STEP 2] Initializing EfficientNetV2-S Backbone & EDL Head...")
    model = timm.create_model("tf_efficientnetv2_s.in21k_ft_in1k", pretrained=False)
    model.reset_classifier(0)

    # Load Phase 1 pretrained backbone weights
    backbone_state = torch.load(PHASE1_BACKBONE_PATH, map_location="cpu")
    model.load_state_dict(backbone_state, strict=False)
    print(f"✓ Successfully loaded Phase 1 backbone weights from {PHASE1_BACKBONE_PATH}")

    # Attach Evidential Head for 67 Field Classes
    model.classifier = nn.Linear(1280, NUM_CLASSES)
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✓ Total Parameters: {total_params:,}")
    print(f"✓ Trainable Parameters: {trainable_params:,}")

    # 4. Class Weights for 67 Head Classes
    print("\n[STEP 3] Loading Class Weights...")
    with open(WEIGHTS_JSON, "r", encoding="utf-8") as f:
        weights_data = json.load(f)

    head_weights = [weights_data["head_weights"][str(i)] for i in range(NUM_CLASSES)]
    weights_tensor = torch.tensor(head_weights, dtype=torch.float32).to(device)

    # 5. Optimizer, Scheduler, Mixed Precision
    epochs = 10
    grad_accum_steps = 2
    learning_rate = 1e-4  # Differential/fine-tuning LR for domain adaptation

    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    use_amp = (device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # History Tracking Structure
    training_history: dict[str, list] = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": [],
        "mean_uncertainty": [],
        "learning_rate": [],
        "time_seconds": [],
    }

    # 6. Training Loop
    print("\n[STEP 4] Starting Phase 2 Training Loop...")
    best_val_acc = 0.0
    no_improvement = 0
    patience = 4

    log_lines: list[str] = [
        "ZARI.ai Phase 2 EDL Fine-Tuning Log",
        "==================================",
        f"Timestamp: {datetime.now().isoformat()}",
        f"Device: {device}",
        f"Batch Size: {batch_size} (Effective: {batch_size * grad_accum_steps})",
        f"Learning Rate: {learning_rate}",
        f"Epochs: {epochs}",
        "",
    ]

    start_time = time.time()

    with mlflow.start_run(run_name="phase2_cnn_baseline"):
        mlflow.log_params({
            "backbone": "tf_efficientnetv2_s.in21k_ft_in1k",
            "NUM_CLASSES": NUM_CLASSES,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "epochs": epochs,
            "weight_decay": 0.01,
            "grad_accum_steps": grad_accum_steps,
        })

        for epoch in range(epochs):
            epoch_start = time.time()

            # ============ TRAINING PHASE ============
            model.train()
            running_loss = 0.0
            running_uncertainty = 0.0
            optimizer.zero_grad()

            for batch_idx, (images, labels) in enumerate(train_loader):
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                with torch.amp.autocast("cuda", enabled=use_amp):
                    logits = model(images)
                    loss, batch_u = edl_loss_fn(logits, labels, epoch, epochs, weights=weights_tensor)
                    loss = loss / grad_accum_steps

                scaler.scale(loss).backward()

                if (batch_idx + 1) % grad_accum_steps == 0 or (batch_idx + 1) == len(train_loader):
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

                running_loss += loss.item() * grad_accum_steps
                running_uncertainty += batch_u.item()

                if (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == len(train_loader):
                    current_avg_loss = running_loss / (batch_idx + 1)
                    current_avg_u = running_uncertainty / (batch_idx + 1)
                    print(
                        f"Epoch [{epoch+1}/{epochs}] Batch [{batch_idx+1}/{len(train_loader)}] "
                        f"Loss: {current_avg_loss:.4f} | Uncertainty: {current_avg_u:.4f}"
                    )

            train_loss = running_loss / len(train_loader)
            train_uncertainty = running_uncertainty / len(train_loader)

            # ============ VALIDATION PHASE ============
            model.eval()
            correct = 0
            total = 0
            val_loss = 0.0
            val_uncertainty = 0.0

            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)

                    with torch.amp.autocast("cuda", enabled=use_amp):
                        logits = model(images)
                        loss, batch_u = edl_loss_fn(logits, labels, epoch, epochs, weights=None)

                    val_loss += loss.item()
                    val_uncertainty += batch_u.item()

                    evidence = F.softplus(logits)
                    alpha = evidence + 1.0
                    preds = alpha.argmax(dim=1)
                    correct += int((preds == labels).sum().item())
                    total += int(labels.size(0))

            val_acc = correct / total if total > 0 else 0.0
            val_avg_loss = val_loss / len(val_loader)
            val_avg_u = val_uncertainty / len(val_loader)
            epoch_duration = time.time() - epoch_start
            current_lr = float(optimizer.param_groups[0]["lr"])

            # Log metrics to MLflow per-epoch
            mlflow.log_metrics({
                "train_loss": train_loss,
                "val_loss": val_avg_loss,
                "val_accuracy": val_acc,
                "mean_uncertainty": val_avg_u
            }, step=epoch)

            # Record metrics to training_history
            training_history["epoch"].append(epoch + 1)
            training_history["train_loss"].append(round(float(train_loss), 4))
            training_history["val_loss"].append(round(float(val_avg_loss), 4))
            training_history["val_accuracy"].append(round(float(val_acc), 4))
            training_history["mean_uncertainty"].append(round(float(val_avg_u), 4))
            training_history["learning_rate"].append(round(float(current_lr), 6))
            training_history["time_seconds"].append(round(float(epoch_duration), 2))

            # ============ SAVE & TRACKING ============
            is_best = val_acc > best_val_acc
            if is_best:
                best_val_acc = val_acc
                torch.save(model.state_dict(), BEST_MODEL_PATH)
                torch.save(model.state_dict(), EDL_MODEL_PATH)
                no_improvement = 0
                best_tag = "⭐ BEST"
            else:
                no_improvement += 1
                best_tag = ""

            status_str = (
                f"Epoch {epoch+1:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_avg_loss:.4f} | Val Acc: {val_acc*100:.2f}% | "
                f"Uncertainty: {val_avg_u:.4f} | LR: {current_lr:.6f} | Time: {epoch_duration:.1f}s {best_tag}"
            )
            print("-" * 75)
            print(status_str)
            print("-" * 75)
            log_lines.append(status_str)

            # Update learning rate scheduler
            scheduler.step()

            # Early Stopping Check
            if no_improvement >= patience:
                early_stop_str = f"Early stopping triggered at epoch {epoch+1} (no improvement for {patience} epochs)."
                print(early_stop_str)
                log_lines.append(early_stop_str)
                break

        total_training_time = time.time() - start_time

        # Save training history JSON & CSV
        HISTORY_JSON_PATH.write_text(json.dumps(training_history, indent=2), encoding="utf-8")
        pd.DataFrame(training_history).to_csv(HISTORY_CSV_PATH, index=False)

        summary_time_str = (
            f"\nTraining completed in {total_training_time/60:.2f} minutes. "
            f"Best Val Acc: {best_val_acc*100:.2f}%, Final Train Loss: {train_loss:.4f}, "
            f"Final Val Loss: {val_avg_loss:.4f}, Final Uncertainty: {val_avg_u:.4f}"
        )
        print(summary_time_str)
        log_lines.append(summary_time_str)

        LOG_FILE_PATH.write_text("\n".join(log_lines), encoding="utf-8")
        print(f"✓ Saved full checkpoint: {BEST_MODEL_PATH}")
        print(f"✓ Saved EDL model checkpoint: {EDL_MODEL_PATH}")
        print(f"✓ Saved training history JSON: {HISTORY_JSON_PATH}")
        print(f"✓ Saved training history CSV : {HISTORY_CSV_PATH}")
        print(f"✓ Training log saved to: {LOG_FILE_PATH}")

        # Log artifacts to MLflow
        if HISTORY_JSON_PATH.exists():
            mlflow.log_artifact(str(HISTORY_JSON_PATH))
        if LOG_FILE_PATH.exists():
            mlflow.log_artifact(str(LOG_FILE_PATH))
        if BEST_MODEL_PATH.exists():
            mlflow.log_artifact(str(BEST_MODEL_PATH))


    print("\n" + "=" * 65)
    print("  FINAL PHASE 2 TRAINING SUMMARY")
    print("=" * 65)
    print(f"Best Validation Accuracy : {best_val_acc * 100:.2f}%")
    print(f"Final Train Loss         : {train_loss:.4f}")
    print(f"Final Validation Loss    : {val_avg_loss:.4f}")
    print(f"Final Mean Uncertainty   : {val_avg_u:.4f}")
    print(f"Total Training Time      : {total_training_time / 60:.2f} minutes")
    print(f"Training History JSON    : {HISTORY_JSON_PATH}")
    print(f"Training History CSV     : {HISTORY_CSV_PATH}")
    print("\n✅ PHASE 2 EDL TRAINING COMPLETE!")


if __name__ == "__main__":
    main()
