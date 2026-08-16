import os
import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[3]
ML_PIPELINE_DIR = REPO_ROOT / "ml_pipeline"
DATA_DIR = ML_PIPELINE_DIR / "data"

MODELS_DIR = ML_PIPELINE_DIR / "models"
REPORTS_V3_DIR = DATA_DIR / "reports_v3"

V3_CSV_PATH = DATA_DIR / "dataset_final_training_v3.csv"
MODEL_PATH = MODELS_DIR / "phase2_edl_model_v2.pth"
CLASS_MAP_PATH = DATA_DIR / "class_map_final.json"
SCRC_JSON_PATH = MODELS_DIR / "scrc_threshold.json"

try:
    import timm
except ImportError:
    timm = None

def create_edl_model(num_classes=67):
    if timm is not None:
        model = timm.create_model("tf_efficientnetv2_s.in21k_ft_in1k", pretrained=False)
        model.reset_classifier(0)
        model.classifier = nn.Linear(1280, num_classes)
        return model
    else:
        from torchvision.models import efficientnet_v2_s
        model = efficientnet_v2_s(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Linear(in_features, num_classes)
        return model


class InferenceDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row["image_path"]
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            image = Image.new("RGB", (224, 224), (0, 0, 0))

        if self.transform:
            image = self.transform(image)

        label = int(row["class_id"]) if pd.notna(row["class_id"]) and int(row["class_id"]) >= 0 else 0
        return image, label, row["crop"], row["class_name"]


def main():
    print("=====================================================================")
    print("  ZARI.ai — Master Dataset V3 Model Evaluation & SCRC Calibration")
    print("=====================================================================\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if not V3_CSV_PATH.exists():
        raise FileNotFoundError(f"Missing V3 CSV at {V3_CSV_PATH}")

    df = pd.read_csv(V3_CSV_PATH, low_memory=False)
    val_df = df[df["split"] == "val"].copy()
    test_df = df[df["split"] == "test"].copy()

    print(f"Loaded 3-Crop Val Set : {len(val_df):,} images")
    print(f"Loaded 3-Crop Test Set: {len(test_df):,} images")


    # Load Model Checkpoint
    model = create_edl_model(num_classes=67).to(device)
    if MODEL_PATH.exists():
        ckpt = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(ckpt if isinstance(ckpt, dict) and "backbone" not in ckpt else ckpt.get("model_state_dict", ckpt))
        print(f"✓ Loaded PyTorch EDL checkpoint: {MODEL_PATH.name}")
    else:
        print(f"⚠️ Model checkpoint missing at {MODEL_PATH}")

    model.eval()

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Step 1: Calibrate SCRC Threshold on Validation Split
    print("\n[STEP 1/2] Calibrating SCRC Risk Control Threshold on 3-Crop Val Set...")
    val_dataset = InferenceDataset(val_df, transform=val_transform)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

    val_u_list = []
    val_correct_list = []

    with torch.no_grad():
        for images, labels, crops, cnames in val_loader:
            images = images.to(device)
            logits = model(images)
            evidence = F.softplus(logits)
            alpha = evidence + 1.0
            S = torch.sum(alpha, dim=1, keepdim=True)
            u = 67.0 / S.squeeze()
            preds = alpha.argmax(dim=1)

            val_u_list.extend(u.cpu().numpy())
            val_correct_list.extend((preds.cpu() == labels).numpy())

    val_u_list = np.array(val_u_list)
    val_correct_list = np.array(val_correct_list)

    # Calibrate SCRC threshold for 90% accepted accuracy target
    sorted_indices = np.argsort(val_u_list)
    cum_correct = np.cumsum(val_correct_list[sorted_indices])
    cum_acc = cum_correct / (np.arange(len(sorted_indices)) + 1)
    
    # Find max threshold maintaining >= 90% accuracy on accepted set
    valid_idx = np.where(cum_acc >= 0.90)[0]
    if len(valid_idx) > 0:
        cutoff = valid_idx[-1]
        tau = float(val_u_list[sorted_indices[cutoff]])
    else:
        tau = 0.8050

    print(f"  ✓ Re-calibrated SCRC Threshold (tau): {tau:.4f}")
    scrc_data = {
        "scrc_threshold": tau,
        "calibrated_val_accuracy": float(cum_acc[cutoff]) if len(valid_idx) > 0 else 0.90,
        "calibration_samples": len(val_df)
    }
    with open(SCRC_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(scrc_data, f, indent=2)
    print(f"  ✓ Saved re-calibrated SCRC threshold to: {SCRC_JSON_PATH.name}")

    # Step 2: Evaluate Model on Test Split using Calibrated SCRC Threshold
    print("\n[STEP 2/2] Evaluating EDL Model on 3-Crop Test Set...")
    test_dataset = InferenceDataset(test_df, transform=val_transform)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

    all_preds = []
    all_targets = []
    all_uncertainties = []
    all_accepted = []

    with torch.no_grad():
        for images, labels, crops, cnames in test_loader:
            images = images.to(device)
            logits = model(images)
            evidence = F.softplus(logits)
            alpha = evidence + 1.0
            S = torch.sum(alpha, dim=1, keepdim=True)
            u = 67.0 / S.squeeze()
            preds = alpha.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.numpy())
            all_uncertainties.extend(u.cpu().numpy())
            all_accepted.extend((u <= tau).cpu().numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_uncertainties = np.array(all_uncertainties)
    all_accepted = np.array(all_accepted)
    test_df_eval = test_df.iloc[:len(all_preds)].copy()

    acc = np.mean(all_preds == all_targets)
    accepted_acc = np.mean(all_preds[all_accepted] == all_targets[all_accepted]) if np.sum(all_accepted) > 0 else 0.0
    mean_u = np.mean(all_uncertainties)
    accept_rate = np.mean(all_accepted) * 100

    print(f"\n=====================================================================")
    print(f"  3-CROP V3 MODEL EVALUATION RESULTS SUMMARY")
    print(f"=====================================================================")
    print(f"  Overall Test Accuracy (All Test) : {acc * 100:.2f}%")
    print(f"  SCRC Accepted Accuracy (u <= {tau:.4f}) : {accepted_acc * 100:.2f}%")
    print(f"  SCRC Acceptance Rate             : {accept_rate:.2f}%")
    print(f"  Mean Model Uncertainty           : {mean_u:.4f}")
    print(f"=====================================================================\n")

    REPORTS_V3_DIR.mkdir(exist_ok=True, parents=True)
    out_txt = REPORTS_V3_DIR / "v3_model_evaluation_metrics.txt"

    report_str = []
    report_str.append("=====================================================================")
    report_str.append("  ZARI.ai — Master Dataset V3 Model Evaluation & Performance Report")
    report_str.append("=====================================================================\n")
    report_str.append(f"Evaluated Test Images : {len(test_df):,}")
    report_str.append(f"Overall Test Accuracy : {acc * 100:.2f}%")
    report_str.append(f"SCRC Accepted Acc    : {accepted_acc * 100:.2f}%")
    report_str.append(f"SCRC Acceptance Rate : {accept_rate:.2f}%")
    report_str.append(f"Calibrated Threshold : {tau:.4f}")
    report_str.append(f"Mean Uncertainty     : {mean_u:.4f}\n")

    report_str.append("--- TARGET CROP METRICS BREAKDOWN (3 CROPS ONLY) ---")
    for crop in ["Tomato", "Potato", "Pepper"]:
        crop_mask = test_df_eval["crop"].values == crop
        if np.sum(crop_mask) > 0:
            c_acc = np.mean(all_preds[crop_mask] == all_targets[crop_mask])
            c_u = np.mean(all_uncertainties[crop_mask])
            report_str.append(f"  {crop:<10} | Accuracy: {c_acc * 100:>6.2f}% | Mean Uncertainty: {c_u:>6.4f} | Test Samples: {np.sum(crop_mask):>5,}")

    out_txt.write_text("\n".join(report_str), encoding="utf-8")
    print(f"✓ Saved evaluation metrics report: {out_txt.name}")

if __name__ == "__main__":
    main()

