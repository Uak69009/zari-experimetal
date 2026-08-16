"""Phase 1 Backbone Pretraining for ZARI.ai (EfficientNetV2-S).

Phase 1 Strategy:
- Train the backbone on ALL 106 merged classes (lab + field data) to learn rich,
  generalizable plant disease visual representations.
- Use Light Augmentation to allow the backbone to learn clean disease features
  first before heavy field noise is introduced in Phase 2.
- Use Weighted CrossEntropyLoss with clipped inverse-frequency weights to handle class imbalance.
- Discard the 106-class linear head after Phase 1, saving ONLY the backbone weights
  (phase1_backbone.pth) for Phase 2 domain adaptation with an Evidential Deep Learning (EDL) head.
- Track complete training history (epoch, train_loss, val_loss, val_accuracy, learning_rate, time_seconds)
  and export as both JSON and CSV.
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
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

try:
    import timm
except ImportError:
    timm = None

# Script and Directory Paths
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"
MODELS_DIR = SCRIPT_DIR / "models"
LOGS_DIR = SCRIPT_DIR / "logs"
INPUT_CSV = DATA_DIR / "dataset_final_training.csv"
WEIGHTS_JSON = DATA_DIR / "class_weights.json"
RAW_ROOT = DATA_DIR / "raw"

BEST_MODEL_PATH = MODELS_DIR / "phase1_best.pth"
BACKBONE_PATH = MODELS_DIR / "phase1_backbone.pth"
LOG_FILE_PATH = LOGS_DIR / "phase1_training_log.txt"
HISTORY_JSON_PATH = LOGS_DIR / "phase1_training_history.json"
HISTORY_CSV_PATH = LOGS_DIR / "phase1_training_history.csv"


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


class PlantDataset(Dataset):
    """Custom PyTorch Dataset for ZARI.ai Phase 1 Pretraining."""

    def __init__(self, df: pd.DataFrame, class_to_idx: dict[str, int], transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.class_to_idx = class_to_idx
        self.image_paths = self.df["image_path"].astype(str).tolist()
        self.labels = [self.class_to_idx[name] for name in self.df["class_name"].astype(str)]

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


def main() -> None:
    print("=" * 65)
    print("  ZARI.ai — PHASE 1 BACKBONE PRETRAINING (EfficientNetV2-S)")
    print("=" * 65)

    if timm is None:
        raise ImportError("timm module is required. Please install timm: pip install timm")

    # Ensure output directories exist
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Missing input dataset CSV at {INPUT_CSV}")
    if not WEIGHTS_JSON.exists():
        raise FileNotFoundError(f"Missing class weights JSON at {WEIGHTS_JSON}")

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing compute device: {device}")
    if device.type == "cuda":
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"VRAM Available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # 1. Load Dataset
    df = pd.read_csv(INPUT_CSV)
    print(f"\n[STEP 1] Loaded dataset: {INPUT_CSV} ({len(df):,} rows)")

    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"].copy()

    # Create mapping for 106 pretrain classes sorted alphabetically
    all_pretrain_classes = sorted(train_df["class_name"].unique().tolist())
    num_classes = len(all_pretrain_classes)
    class_to_idx = {name: idx for idx, name in enumerate(all_pretrain_classes)}

    print(f"✓ Training samples  : {len(train_df):,}")
    print(f"✓ Validation samples: {len(val_df):,}")
    print(f"✓ Total Classes     : {num_classes}")

    # 2. Transforms (Light Augmentation for Feature Extraction)
    # WHY LIGHT AUGMENTATION IN PHASE 1:
    # Heavy field noise early in training can hinder feature learning.
    # Phase 1 uses light augmentation (horizontal flips, mild color jitter) to let
    # the backbone learn robust fundamental features before heavy field noise is applied in Phase 2.
    train_transform = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = PlantDataset(train_df, class_to_idx, transform=train_transform)
    val_dataset = PlantDataset(val_df, class_to_idx, transform=val_transform)

    batch_size = 64 if device.type == "cuda" else 16
    num_workers = min(8, os.cpu_count() or 4)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=(device.type == "cuda")
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=(device.type == "cuda")
    )

    print(f"✓ Batch Size: {batch_size} (Num Workers: {num_workers})")

    # 3. Model Setup (ImageNet Pretrained EfficientNetV2-S)
    # WHY ImageNet PRETRAINING MATTERS:
    # ImageNet pretraining provides strong lower-level visual feature extractors (edges, textures, shapes)
    # learned from 1.2M images, enabling faster convergence and preventing overfitting on specialized leaf domains.
    print("\n[STEP 2] Initializing EfficientNetV2-S Model...")
    model = timm.create_model("tf_efficientnetv2_s.in21k_ft_in1k", pretrained=True)
    model.reset_classifier(0)  # Remove original ImageNet 1000-class head
    model.classifier = nn.Linear(1280, num_classes)
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✓ Total Parameters: {total_params:,}")
    print(f"✓ Trainable Parameters: {trainable_params:,}")

    # 4. Class Weights & Loss Function
    print("\n[STEP 3] Loading Class Weights & Setting Criterion...")
    with open(WEIGHTS_JSON, "r", encoding="utf-8") as f:
        weights_data = json.load(f)

    # Match weights to sorted pretrain class names
    pretrain_weights = [weights_data["pretrain_weights_by_name"][name] for name in all_pretrain_classes]
    weights_tensor = torch.tensor(pretrain_weights, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights_tensor)

    # 5. Optimizer, Scheduler, Mixed Precision, and Accumulation Setup
    # WHY GRADIENT ACCUMULATION:
    # Accumulating gradients over 2 steps simulates an effective batch size of 128 (64 x 2),
    # producing smoother gradient estimates without requiring additional VRAM.
    epochs = 10
    grad_accum_steps = 2
    learning_rate = 3e-4

    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    use_amp = (device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # Training History Structure
    training_history: dict[str, list] = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": [],
        "learning_rate": [],
        "time_seconds": [],
    }

    # 6. Training Loop
    print("\n[STEP 4] Starting Phase 1 Training Loop...")
    best_val_acc = 0.0
    no_improvement = 0
    patience = 3

    log_lines: list[str] = [
        "ZARI.ai Phase 1 Training Log",
        "============================",
        f"Timestamp: {datetime.now().isoformat()}",
        f"Device: {device}",
        f"Batch Size: {batch_size} (Effective: {batch_size * grad_accum_steps})",
        f"Learning Rate: {learning_rate}",
        f"Epochs: {epochs}",
        "",
    ]

    start_time = time.time()

    for epoch in range(epochs):
        epoch_start = time.time()

        # ============ TRAINING PHASE ============
        model.train()
        running_loss = 0.0
        optimizer.zero_grad()

        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, labels)
                loss = loss / grad_accum_steps

            scaler.scale(loss).backward()

            if (batch_idx + 1) % grad_accum_steps == 0 or (batch_idx + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            running_loss += loss.item() * grad_accum_steps

            if (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == len(train_loader):
                current_avg_loss = running_loss / (batch_idx + 1)
                print(
                    f"Epoch [{epoch+1}/{epochs}] Batch [{batch_idx+1}/{len(train_loader)}] Loss: {current_avg_loss:.4f}"
                )

        train_loss = running_loss / len(train_loader)

        # ============ VALIDATION PHASE ============
        model.eval()
        correct = 0
        total = 0
        val_loss = 0.0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                with torch.amp.autocast("cuda", enabled=use_amp):
                    logits = model(images)
                    loss = criterion(logits, labels)

                val_loss += loss.item()
                preds = logits.argmax(dim=1)
                correct += int((preds == labels).sum().item())
                total += int(labels.size(0))

        val_acc = correct / total if total > 0 else 0.0
        val_avg_loss = val_loss / len(val_loader)
        epoch_duration = time.time() - epoch_start
        current_lr = float(optimizer.param_groups[0]["lr"])

        # Record metrics to training_history
        training_history["epoch"].append(epoch + 1)
        training_history["train_loss"].append(round(float(train_loss), 4))
        training_history["val_loss"].append(round(float(val_avg_loss), 4))
        training_history["val_accuracy"].append(round(float(val_acc), 4))
        training_history["learning_rate"].append(round(float(current_lr), 6))
        training_history["time_seconds"].append(round(float(epoch_duration), 2))

        # ============ SAVE & TRACKING ============
        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            no_improvement = 0
            best_tag = "⭐ BEST"
        else:
            no_improvement += 1
            best_tag = ""

        status_str = (
            f"Epoch {epoch+1:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_avg_loss:.4f} | Val Acc: {val_acc*100:.2f}% | "
            f"LR: {current_lr:.6f} | Time: {epoch_duration:.1f}s {best_tag}"
        )
        print("-" * 65)
        print(status_str)
        print("-" * 65)
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
        f"Best Val Acc: {best_val_acc*100:.2f}%, Final Train Loss: {train_loss:.4f}, Final Val Loss: {val_avg_loss:.4f}"
    )
    print(summary_time_str)
    log_lines.append(summary_time_str)

    # 7. Extract & Save Backbone Only
    # WHY DISCARD THE HEAD AFTER PHASE 1:
    # Phase 1 uses a temporary 106-class Linear head to train the backbone on general feature extraction.
    # After Phase 1, this classification head is discarded because Phase 2 will attach a specialized
    # Evidential Deep Learning (EDL) head specifically tailored to the 67 target field classes.
    print("\n[STEP 5] Extracting and Saving Backbone Only for Phase 2...")
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))

    # Strip the classification head parameters
    backbone_state_dict = {
        k: v for k, v in model.state_dict().items() if not k.startswith("classifier.")
    }
    torch.save(backbone_state_dict, BACKBONE_PATH)
    print(f"✓ Saved full checkpoint: {BEST_MODEL_PATH}")
    print(f"✓ Saved backbone-only weights: {BACKBONE_PATH}")
    print(f"✓ Saved training history JSON: {HISTORY_JSON_PATH}")
    print(f"✓ Saved training history CSV : {HISTORY_CSV_PATH}")

    # Verify backbone feature extractor output dimension
    model.reset_classifier(0)
    model.eval()
    dummy_input = torch.randn(1, 3, 384, 384).to(device)
    with torch.no_grad():
        with torch.amp.autocast("cuda", enabled=use_amp):
            features = model(dummy_input)

    print(f"✓ Verified backbone output feature dimension: {features.shape}")
    assert features.shape == (1, 1280), f"Expected backbone feature shape (1, 1280), got {features.shape}"

    LOG_FILE_PATH.write_text("\n".join(log_lines), encoding="utf-8")
    print(f"✓ Training log saved to: {LOG_FILE_PATH}")

    print("\n" + "=" * 65)
    print("  FINAL TRAINING SUMMARY")
    print("=" * 65)
    print(f"Best Validation Accuracy : {best_val_acc * 100:.2f}%")
    print(f"Final Train Loss         : {train_loss:.4f}")
    print(f"Final Validation Loss    : {val_avg_loss:.4f}")
    print(f"Total Training Time      : {total_training_time / 60:.2f} minutes")
    print(f"Training History JSON    : {HISTORY_JSON_PATH}")
    print(f"Training History CSV     : {HISTORY_CSV_PATH}")
    print("\n✅ PHASE 1 TRAINING COMPLETE! Backbone is ready for Phase 2 fine-tuning.")


if __name__ == "__main__":
    main()
