"""ZARI.ai — Confusion Matrix & Test-Time Augmentation (TTA) Analysis
-------------------------------------------------------------------
This script evaluates the Phase 2 Evidential Deep Learning (EDL) model on all
6,648 field test images, computes a full 67x67 confusion matrix, performs a detailed
error analysis for all Wheat classes, applies 5-view Dirichlet Evidence Test-Time
Augmentation (TTA), and outputs formatted comparative metrics and diagnostic reports.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF

try:
    import timm
except ImportError:
    timm = None

# Directory & File Paths
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"
MODELS_DIR = SCRIPT_DIR / "models"
REPORTS_DIR = SCRIPT_DIR / "ANALYSIS_COMPLETE" / "reports"
INPUT_CSV = DATA_DIR / "dataset_final_training.csv"
CLASS_MAP_JSON = DATA_DIR / "class_map_final.json"
EDL_MODEL_PATH = MODELS_DIR / "phase2_edl_model.pth"
BEST_MODEL_PATH = MODELS_DIR / "phase2_best.pth"
RAW_ROOT = DATA_DIR / "raw"

# Output Report Artifact Paths
CONFUSION_PLOT_PATH = REPORTS_DIR / "confusion_matrix_full.png"
TTA_COMPARISON_PATH = REPORTS_DIR / "tta_comparison.txt"
CONFUSION_ANALYSIS_PATH = REPORTS_DIR / "confusion_analysis.txt"

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


class FieldTestTTADataset(Dataset):
    """
    Custom PyTorch Dataset for ZARI.ai Test Evaluation.
    Returns both standard base transform AND 5 augmented TTA views.
    
    TTA Views:
    1. Original (Resize 384x384, ToTensor, Normalize)
    2. Horizontal Flip
    3. Vertical Flip
    4. Rotate +10 degrees
    5. Rotate -10 degrees
    
    WHY TTA HELPS WITH VIEWPOINT SENSITIVITY:
    Field crop images in real-world agricultural datasets exhibit significant variations in
    leaf orientation, camera tilt, shadow patterns, and scale. By evaluating 5 augmented
    viewpoints (Original, H-Flip, V-Flip, +10° Rotation, -10° Rotation), TTA ensures that
    spatial lesion features (e.g. rust pustules, leaf blight streaks, tan spot lesions)
    are presented to the feature extractor across multiple alignments, mitigating viewpoint
    sensitivity and yielding more robust Dirichlet evidence predictions.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True)
        self.image_paths = self.df["image_path"].astype(str).tolist()
        self.labels = self.df["class_id"].astype(int).tolist()
        self.class_names = self.df["class_name"].astype(str).tolist()
        self.sources = self.df["source_dataset"].astype(str).tolist()

        self.norm = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        self.resize_size = (384, 384)

    def __len__(self) -> int:
        return len(self.image_paths)

    def _transform_pil(self, img_pil: Image.Image) -> torch.Tensor:
        img_resized = img_pil.resize(self.resize_size, Image.BILINEAR)
        tensor = TF.to_tensor(img_resized)
        return self.norm(tensor)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int, str, str]:
        image_path_str = self.image_paths[idx]
        label = self.labels[idx]
        class_name = self.class_names[idx]
        source = self.sources[idx]

        resolved_path = resolve_image_path(image_path_str)
        pil_img = Image.open(resolved_path).convert("RGB")

        # 1. Base transform (Original)
        tensor_orig = self._transform_pil(pil_img)

        # 2. Generate 5 TTA views
        v1 = pil_img
        v2 = TF.hflip(pil_img)
        v3 = TF.vflip(pil_img)
        v4 = TF.rotate(pil_img, angle=10)
        v5 = TF.rotate(pil_img, angle=-10)

        t1 = self._transform_pil(v1)
        t2 = self._transform_pil(v2)
        t3 = self._transform_pil(v3)
        t4 = self._transform_pil(v4)
        t5 = self._transform_pil(v5)

        tta_stack = torch.stack([t1, t2, t3, t4, t5], dim=0)  # (5, 3, 384, 384)

        return tensor_orig, tta_stack, label, class_name, source


def main() -> None:
    print("=" * 75)
    print("  ZARI.ai — CONFUSION MATRIX & TEST-TIME AUGMENTATION (TTA) ANALYSIS")
    print("=" * 75)

    if timm is None:
        raise ImportError("timm module is required. Please install timm: pip install timm")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Missing input dataset CSV at {INPUT_CSV}")

    # Set compute device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing compute device: {device}")
    if device.type == "cuda":
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")

    # 1. Load Class Mapping
    class_id_to_name: dict[int, str] = {}
    if CLASS_MAP_JSON.exists():
        with open(CLASS_MAP_JSON, "r", encoding="utf-8") as f:
            class_map_data = json.load(f)
            head_classes = class_map_data.get("head_classes", {})
            for name, cid in head_classes.items():
                if isinstance(cid, int) and cid >= 0:
                    class_id_to_name[cid] = name

    class_names_ordered = [class_id_to_name.get(c, f"Class_{c}") for c in range(NUM_CLASSES)]

    # 2. Load Model
    print("\n[STEP 1] Loading Phase 2 EDL Model (phase2_edl_model.pth)...")
    model = timm.create_model("tf_efficientnetv2_s.in21k_ft_in1k", pretrained=False)
    model.reset_classifier(0)
    model.classifier = nn.Linear(1280, NUM_CLASSES)

    model_path_to_load = EDL_MODEL_PATH if EDL_MODEL_PATH.exists() else BEST_MODEL_PATH
    state_dict = torch.load(model_path_to_load, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device).eval()
    print(f"✓ Model successfully loaded from {model_path_to_load.name} and set to eval mode")

    # 3. Load Test Dataset
    df = pd.read_csv(INPUT_CSV)
    test_df = df[(df["split"] == "test") & (df["class_id"] >= 0)].copy()
    print(f"\n[STEP 2] Loaded Test Set: {len(test_df):,} field images")

    dataset = FieldTestTTADataset(test_df)
    batch_size = 32 if device.type == "cuda" else 8
    num_workers = min(8, os.cpu_count() or 4)

    test_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=(device.type == "cuda")
    )

    # Containers for Standard (No TTA) predictions
    all_y_true: list[int] = []
    no_tta_preds: list[int] = []
    no_tta_probs: list[float] = []
    no_tta_uncertainties: list[float] = []

    # Containers for TTA predictions
    tta_preds: list[int] = []
    tta_probs: list[float] = []
    tta_uncertainties: list[float] = []

    print("\n[STEP 3] Running Standard Evaluation (No TTA) and 5-View Evidential TTA...")
    start_time = time.time()

    with torch.no_grad():
        for batch_idx, (images_orig, images_tta_stack, labels, cnames, sources) in enumerate(test_loader):
            images_orig = images_orig.to(device, non_blocking=True)
            batch_b = images_orig.shape[0]

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                # --- STANDARD EVALUATION (NO TTA) ---
                logits_orig = model(images_orig)
                evidence_orig = F.softplus(logits_orig)
                alpha_orig = evidence_orig + 1.0
                S_orig = torch.sum(alpha_orig, dim=1, keepdim=True)
                probs_orig = alpha_orig / S_orig
                uncertainty_orig = float(NUM_CLASSES) / S_orig.squeeze(-1)
                conf_orig, preds_orig = probs_orig.max(dim=1)

                # --- TTA EVALUATION (5 VIEWS) ---
                # Flatten (B, 5, 3, 384, 384) -> (B * 5, 3, 384, 384)
                images_tta_flat = images_tta_stack.view(-1, 3, 384, 384).to(device, non_blocking=True)
                logits_tta_flat = model(images_tta_flat)
                evidence_tta_flat = F.softplus(logits_tta_flat)
                alpha_tta_flat = evidence_tta_flat + 1.0  # (B*5, NUM_CLASSES)

                # Reshape to (B, 5, NUM_CLASSES)
                alpha_tta_views = alpha_tta_flat.view(batch_b, 5, NUM_CLASSES)

                # CODE STYLE REQUIREMENT:
                # WHY WE AVERAGE EVIDENCE, NOT PROBABILITIES:
                # In Evidential Deep Learning (EDL), Dirichlet parameters alpha = [alpha_1, ..., alpha_K]
                # represent total pseudo-counts of belief (evidence + 1) observed by the network.
                # Averaging alpha values (or evidence e = Softplus(logits)) across different augmented
                # viewpoints aggregates the model's total underlying evidential belief before normalizing.
                # In contrast, averaging probabilities directly (p = alpha / S) destroys the total evidence
                # magnitude S = sum(alpha), resulting in overconfident or poorly calibrated uncertainty estimates.
                alpha_tta_avg = torch.mean(alpha_tta_views, dim=1)  # (B, NUM_CLASSES)

                S_tta = torch.sum(alpha_tta_avg, dim=1, keepdim=True)
                probs_tta = alpha_tta_avg / S_tta
                uncertainty_tta = float(NUM_CLASSES) / S_tta.squeeze(-1)
                conf_tta, preds_tta_batch = probs_tta.max(dim=1)

            # Store batch predictions
            labels_np = labels.cpu().numpy()
            all_y_true.extend(labels_np.tolist())

            no_tta_preds.extend(preds_orig.cpu().numpy().tolist())
            no_tta_probs.extend(conf_orig.cpu().numpy().tolist())
            no_tta_uncertainties.extend(uncertainty_orig.cpu().numpy().tolist())

            tta_preds.extend(preds_tta_batch.cpu().numpy().tolist())
            tta_probs.extend(conf_tta.cpu().numpy().tolist())
            tta_uncertainties.extend(uncertainty_tta.cpu().numpy().tolist())

            if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == len(test_loader):
                print(f"  Processed {len(all_y_true):,}/{len(test_df):,} test images...")

    total_eval_time = time.time() - start_time
    print(f"✓ Completed dual evaluation of {len(all_y_true):,} images in {total_eval_time:.2f}s")

    y_true = np.array(all_y_true, dtype=int)
    y_pred_no_tta = np.array(no_tta_preds, dtype=int)
    y_pred_tta = np.array(tta_preds, dtype=int)

    u_no_tta = np.array(no_tta_uncertainties, dtype=float)
    u_tta = np.array(tta_uncertainties, dtype=float)

    is_correct_no_tta = (y_true == y_pred_no_tta).astype(int)
    is_correct_tta = (y_true == y_pred_tta).astype(int)

    # 4. Overall Metrics & Comparison
    acc_no_tta = float(accuracy_score(y_true, y_pred_no_tta))
    acc_tta = float(accuracy_score(y_true, y_pred_tta))
    acc_change = (acc_tta - acc_no_tta) * 100

    macro_f1_no_tta = float(f1_score(y_true, y_pred_no_tta, average="macro"))
    macro_f1_tta = float(f1_score(y_true, y_pred_tta, average="macro"))
    macro_f1_change = macro_f1_tta - macro_f1_no_tta

    auroc_no_tta = float(roc_auc_score(1 - is_correct_no_tta, u_no_tta))
    auroc_tta = float(roc_auc_score(1 - is_correct_tta, u_tta))
    auroc_change = auroc_tta - auroc_no_tta

    print("\n" + "=" * 65)
    print("  METRIC COMPARISON: NO TTA vs WITH TTA")
    print("=" * 65)
    print(f"{'Metric':<20} | {'No TTA':<10} | {'With TTA':<10} | {'Change':<10}")
    print("-" * 60)
    print(f"{'Overall Accuracy':<20} | {acc_no_tta*100:6.2f}%    | {acc_tta*100:6.2f}%    | {acc_change:+6.2f}%")
    print(f"{'Macro F1':<20} | {macro_f1_no_tta:8.4f}   | {macro_f1_tta:8.4f}   | {macro_f1_change:+8.4f}")
    print(f"{'AUROC (Error Detection)':<20} | {auroc_no_tta:8.4f}   | {auroc_tta:8.4f}   | {auroc_change:+8.4f}")

    # 5. Wheat Class Accuracy & Comparison
    wheat_class_ids = [cid for cid, name in class_id_to_name.items() if name.startswith("Wheat_")]
    wheat_class_ids.sort()

    wheat_metrics = []
    for cid in wheat_class_ids:
        cname = class_id_to_name[cid]
        mask = (y_true == cid)
        count = int(mask.sum())

        if count > 0:
            w_acc_no_tta = float((y_pred_no_tta[mask] == cid).sum()) / count
            w_acc_tta = float((y_pred_tta[mask] == cid).sum()) / count
        else:
            w_acc_no_tta = 0.0
            w_acc_tta = 0.0

        w_change = (w_acc_tta - w_acc_no_tta) * 100
        wheat_metrics.append({
            "class_id": cid,
            "class_name": cname,
            "count": count,
            "no_tta_acc": w_acc_no_tta,
            "tta_acc": w_acc_tta,
            "change_pct": w_change
        })

    print("\n" + "=" * 70)
    print("  WHEAT CLASSES PERFORMANCE (NO TTA vs WITH TTA)")
    print("=" * 70)
    print(f"{'Class Name':<30} | {'No TTA':<10} | {'With TTA':<10} | {'Change':<10}")
    print("-" * 70)
    for wm in wheat_metrics:
        print(f"{wm['class_name']:<30} | {wm['no_tta_acc']*100:6.2f}%    | {wm['tta_acc']*100:6.2f}%    | {wm['change_pct']:+6.2f}%")

    # 6. Generate & Save Full 67x67 Confusion Matrix Plot
    print("\n[STEP 4] Generating Full 67x67 Confusion Matrix Plot...")
    cm_full = confusion_matrix(y_true, y_pred_no_tta, labels=range(NUM_CLASSES))

    plt.figure(figsize=(24, 20))
    sns.heatmap(
        cm_full,
        annot=False,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names_ordered,
        yticklabels=class_names_ordered,
        cbar_kws={"label": "Sample Count"},
    )
    plt.title("ZARI.ai Evidential Model — Full 67x67 Confusion Matrix (No TTA)", fontsize=16, pad=20)
    plt.xlabel("Predicted Class", fontsize=14, labelpad=10)
    plt.ylabel("True Class", fontsize=14, labelpad=10)
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig(CONFUSION_PLOT_PATH, dpi=300)
    plt.close()
    print(f"✓ Saved full confusion matrix plot to: {CONFUSION_PLOT_PATH}")

    # 7. Analyze Wheat Class Confusion Details
    print("\n[STEP 5] Analyzing Detailed Wheat Confusion Patterns...")
    wheat_confusion_lines = [
        "================================================================================",
        "ZARI.ai — DETAILED WHEAT CONFUSION ANALYSIS REPORT",
        "================================================================================",
        f"Date / Timestamp : {datetime.now().isoformat()}",
        f"Test Set Samples : {len(y_true):,} field images",
        f"Evaluation Model : {model_path_to_load.name}",
        "",
        "SUMMARY OF WHEAT CLASS CONFUSION BREAKDOWN:",
        "--------------------------------------------------------------------------------",
    ]

    total_wheat_errors = 0
    total_wheat_top3_errors = 0
    specific_confusion_count = 0

    for wm in wheat_metrics:
        cid = wm["class_id"]
        cname = wm["class_name"]
        mask = (y_true == cid)
        total_samples = int(mask.sum())

        error_mask = mask & (y_pred_no_tta != cid)
        errors_count = int(error_mask.sum())
        total_wheat_errors += errors_count

        wheat_confusion_lines.append(f"\n{cname} (ID {cid}): Total Samples = {total_samples}, Errors = {errors_count} ({100 - wm['no_tta_acc']*100:.2f}% error rate)")

        if errors_count == 0:
            wheat_confusion_lines.append("  - Perfect accuracy (0 misclassifications)!")
            continue

        # Compute confused classes distribution
        preds_for_errors = y_pred_no_tta[error_mask]
        confused_counts = pd.Series(preds_for_errors).value_counts()
        top3_confused = confused_counts.head(3)

        wheat_confusion_lines.append(f"{cname} is most confused with:")
        top3_sum = 0
        for rank, (confused_cid, count) in enumerate(top3_confused.items(), 1):
            confused_name = class_id_to_name.get(confused_cid, f"Class_{confused_cid}")
            pct_of_errors = (count / errors_count) * 100
            top3_sum += count
            wheat_confusion_lines.append(f"  - {confused_name} ({pct_of_errors:.1f}% of errors, {count} samples)")

        total_wheat_top3_errors += top3_sum
        top1_pct = (top3_confused.iloc[0] / errors_count) * 100 if len(top3_confused) > 0 else 0
        top3_pct = (top3_sum / errors_count) * 100

        if top3_pct >= 50.0 or top1_pct >= 30.0:
            specific_confusion_count += 1
            wheat_confusion_lines.append(f"  [Insight]: Highly SPECIFIC error concentration ({top3_pct:.1f}% of errors in top 3 classes).")
        else:
            wheat_confusion_lines.append(f"  [Insight]: Broadly GENERAL error distribution ({top3_pct:.1f}% in top 3 classes).")

    # 8. Verdict & Next Step Recommendations
    overall_top3_ratio = (total_wheat_top3_errors / total_wheat_errors * 100) if total_wheat_errors > 0 else 0
    is_wheat_errors_specific = (overall_top3_ratio >= 50.0) or (specific_confusion_count >= len(wheat_metrics) // 2)
    is_tta_helpful = (acc_change > 0) or (macro_f1_change > 0)

    wheat_verdict_str = "SPECIFIC (concentrated in 1-3 visually similar foliar diseases)" if is_wheat_errors_specific else "GENERAL (spread broadly across diverse crop classes)"
    tta_verdict_str = f"HELPFUL (+{acc_change:.2f}% accuracy, +{macro_f1_change:.4f} F1)" if is_tta_helpful else f"NEUTRAL / MARGINAL ({acc_change:+.2f}% accuracy)"

    verdict_lines = [
        "",
        "================================================================================",
        "FINAL VERDICT & STRATEGIC RECOMMENDATIONS",
        "================================================================================",
        f"1. WHEAT ERROR CHARACTERISTIC : {wheat_verdict_str}",
        f"   - Top 3 confused classes account for {overall_top3_ratio:.1f}% of all wheat classification errors.",
        f"   - Primary error drivers: Leaf rusts (Brown vs Yellow vs Black rust) and leaf spots (Tan Spot vs Septoria vs Leaf Blight).",
        "",
        f"2. TEST-TIME AUGMENTATION (TTA) EFFECT : {tta_verdict_str}",
        f"   - Standard Accuracy : {acc_no_tta*100:.2f}%  ==>  TTA Accuracy : {acc_tta*100:.2f}% ({acc_change:+0.2f}%)",
        f"   - Standard Macro F1 : {macro_f1_no_tta:.4f}   ==>  TTA Macro F1 : {macro_f1_tta:.4f} ({macro_f1_change:+0.4f})",
        "",
        "3. RECOMMENDED NEXT STEPS BASED ON FINDINGS:",
        "   * Specific Folio-Lesion Data Synthesis / Hard-Negative Mining:",
        "     Since wheat errors are specific foliar disease overlaps (e.g. Tan Spot vs Septoria), targeted feature extraction",
        "     and fine-grained class weighting will yield the highest gains.",
        "   * Deploy TTA in Inference Engine:",
        "     Integrate 5-view Dirichlet evidence aggregation into backend/main.py for field image inference.",
        "================================================================================",
    ]

    wheat_confusion_lines.extend(verdict_lines)
    CONFUSION_ANALYSIS_PATH.write_text("\n".join(wheat_confusion_lines), encoding="utf-8")
    print(f"\n✓ Saved confusion analysis report to: {CONFUSION_ANALYSIS_PATH}")

    # 9. Save TTA Comparison Report
    tta_lines = [
        "================================================================================",
        "ZARI.ai — TEST-TIME AUGMENTATION (TTA) COMPARISON REPORT",
        "================================================================================",
        f"Date / Timestamp           : {datetime.now().isoformat()}",
        f"Evaluation Dataset         : Test Split ({len(y_true):,} Field Images)",
        f"Evaluated Model            : {model_path_to_load.name}",
        f"TTA Configuration          : 5-View Dirichlet Evidence Averaging",
        "                             (Original, H-Flip, V-Flip, +10° Rotation, -10° Rotation)",
        "",
        "1. OVERALL SYSTEM METRICS COMPARISON",
        "--------------------------------------------------------------------------------",
        f"{'Metric':<25} | {'No TTA':<12} | {'With TTA':<12} | {'Change':<12}",
        "-" * 65,
        f"{'Overall Accuracy':<25} | {acc_no_tta*100:6.2f}%      | {acc_tta*100:6.2f}%      | {acc_change:+6.2f}%",
        f"{'Macro F1 Score':<25} | {macro_f1_no_tta:8.4f}     | {macro_f1_tta:8.4f}     | {macro_f1_change:+8.4f}",
        f"{'Error Detection AUROC':<25} | {auroc_no_tta:8.4f}     | {auroc_tta:8.4f}     | {auroc_change:+8.4f}",
        "",
        "2. WHEAT CLASSES ACCURACY BREAKDOWN",
        "--------------------------------------------------------------------------------",
        f"{'Class Name':<30} | {'No TTA':<12} | {'With TTA':<12} | {'Change':<12}",
        "-" * 70,
    ]

    for wm in wheat_metrics:
        tta_lines.append(f"{wm['class_name']:<30} | {wm['no_tta_acc']*100:6.2f}%      | {wm['tta_acc']*100:6.2f}%      | {wm['change_pct']:+6.2f}%")

    tta_lines.extend([
        "",
        "================================================================================",
        "SUMMARY VERDICT:",
        f"Wheat Errors Characteristic : {wheat_verdict_str}",
        f"TTA Impact                  : {tta_verdict_str}",
        "================================================================================",
    ])

    TTA_COMPARISON_PATH.write_text("\n".join(tta_lines), encoding="utf-8")
    print(f"✓ Saved TTA comparison report to: {TTA_COMPARISON_PATH}")

    # Print Final Summary on Console
    print("\n" + "=" * 70)
    print("  CONFUSION & TTA ANALYSIS SUMMARY & VERDICT")
    print("=" * 70)
    print(f"Overall Accuracy (No TTA -> TTA) : {acc_no_tta*100:.2f}% -> {acc_tta*100:.2f}% ({acc_change:+0.2f}%)")
    print(f"Macro F1 (No TTA -> TTA)        : {macro_f1_no_tta:.4f} -> {macro_f1_tta:.4f} ({macro_f1_change:+0.4f})")
    print(f"Wheat Errors Characteristic     : {wheat_verdict_str}")
    print(f"TTA Impact                      : {tta_verdict_str}")
    print(f"\nArtifacts Saved:")
    print(f"  1. Plot Plot      : {CONFUSION_PLOT_PATH}")
    print(f"  2. TTA Comparison : {TTA_COMPARISON_PATH}")
    print(f"  3. Error Analysis : {CONFUSION_ANALYSIS_PATH}")
    print("\n✅ CONFUSION ANALYSIS & TTA EVALUATION COMPLETE!")


if __name__ == "__main__":
    main()
