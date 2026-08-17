import os
import json
import pandas as pd
import numpy as np
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
DATA_DIR = REPO_ROOT / "ml_pipeline" / "data"
REPORTS_V3_DIR = DATA_DIR / "reports_v3"
CSV_PATH = DATA_DIR / "dataset_final_training_v3_clean.csv"
FALLBACK_CSV_PATH = DATA_DIR / "dataset_3crop_final.csv"
WEIGHTS_JSON_PATH = DATA_DIR / "class_weights_v3.json"
REPORT_MD_PATH = REPORTS_V3_DIR / "class_imbalance_report.md"

def main():
    print("=====================================================================")
    print("  ZARI.ai — Class Imbalance Analysis & Class Weights Computation")
    print("=====================================================================\n")

    REPORTS_V3_DIR.mkdir(parents=True, exist_ok=True)

    # 1. LOAD DATASET
    target_csv = CSV_PATH if CSV_PATH.exists() else FALLBACK_CSV_PATH
    if not target_csv.exists():
        raise FileNotFoundError(f"Dataset manifest missing at {target_csv}")

    df = pd.read_csv(target_csv, low_memory=False)
    print(f"1. LOAD DATASET:")
    print(f"   Manifest File: {target_csv.relative_to(REPO_ROOT)}")
    print(f"   Total: {len(df):,} images")

    crops = ["Tomato", "Potato", "Pepper"]

    # 2. PER-CROP COUNTS
    print("\n2. PER-CROP COUNTS:")
    crop_stats = {}
    for crop in crops:
        crop_df = df[df["crop"] == crop]
        n_imgs = len(crop_df)
        n_classes = crop_df["class_name"].nunique()
        tr_cnt = (crop_df["split"] == "train").sum()
        va_cnt = (crop_df["split"] == "val").sum()
        te_cnt = (crop_df["split"] == "test").sum()
        
        crop_stats[crop] = {
            "total": n_imgs,
            "classes": n_classes,
            "train": tr_cnt,
            "val": va_cnt,
            "test": te_cnt
        }
        
        print(f"\n{crop}: {n_imgs:,} images")
        print(f"  Classes: {n_classes}")
        print(f"  Train: {tr_cnt:,}")
        print(f"  Val: {va_cnt:,}")
        print(f"  Test: {te_cnt:,}")

    # 3. PER-CLASS COUNTS
    print("\n3. PER-CLASS COUNTS:")
    per_class_summary = {}
    for crop in crops:
        crop_df = df[df["crop"] == crop]
        print(f"\n=== {crop} Classes ===")
        class_counts = crop_df["class_name"].value_counts().sort_values(ascending=True)
        per_class_summary[crop] = class_counts
        for cls, cnt in class_counts.items():
            print(f"  {cls:<40}: {cnt:>5,}")

    # 4. IMBALANCE STATS PER CROP
    print("\n4. IMBALANCE STATS PER CROP:")
    imbalance_stats = {}
    for crop in crops:
        crop_df = df[df["crop"] == crop]
        counts = crop_df["class_name"].value_counts()
        c_min = int(counts.min())
        c_max = int(counts.max())
        c_mean = float(counts.mean())
        c_ratio = float(c_max / c_min) if c_min > 0 else 0.0

        imbalance_stats[crop] = {
            "min": c_min,
            "max": c_max,
            "mean": round(c_mean, 1),
            "ratio": round(c_ratio, 1)
        }

        print(f"\n{crop}:")
        print(f"  Min: {c_min:,}")
        print(f"  Max: {c_max:,}")
        print(f"  Mean: {c_mean:.1f}")
        print(f"  Ratio: {c_ratio:.1f}x")

    # 5. COMPUTE CLASS WEIGHTS
    # Formula: w_i = N / (K * n_i), normalized + clipped [0.1, 10.0]
    print("\n5. COMPUTE CLASS WEIGHTS:")
    weights_dict = {
        "per_crop": {},
        "global": {}
    }

    # Per-Crop Weights
    for crop in crops:
        crop_df = df[df["crop"] == crop]
        counts = crop_df["class_name"].value_counts()
        N = len(crop_df)
        K = len(counts)

        crop_w = {}
        for cls, n_i in counts.items():
            raw_w = N / (K * n_i)
            crop_w[cls] = raw_w

        # Normalize so mean weight = 1.0
        w_values = np.array(list(crop_w.values()))
        w_normalized = w_values / np.mean(w_values)
        w_clipped = np.clip(w_normalized, 0.1, 10.0)

        for (cls, _), w_val in zip(counts.items(), w_clipped):
            crop_w[cls] = round(float(w_val), 4)

        weights_dict["per_crop"][crop] = crop_w

    # Global Weights
    global_counts = df["class_name"].value_counts()
    N_global = len(df)
    K_global = len(global_counts)
    global_w = {}
    for cls, n_i in global_counts.items():
        raw_w = N_global / (K_global * n_i)
        global_w[cls] = raw_w

    w_g_values = np.array(list(global_w.values()))
    w_g_norm = w_g_values / np.mean(w_g_values)
    w_g_clipped = np.clip(w_g_norm, 0.1, 10.0)

    for (cls, _), w_val in zip(global_counts.items(), w_g_clipped):
        global_w[cls] = round(float(w_val), 4)

    weights_dict["global"] = global_w

    with open(WEIGHTS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(weights_dict, f, indent=2)
    print(f"   Saved Class Weights JSON: {WEIGHTS_JSON_PATH.relative_to(REPO_ROOT)}")

    # 6. SAVE REPORT
    report_md = f"""# ZARI.ai — Class Imbalance & Class Weights Analysis Report (V3 Dataset)

**Analysis Date**: August 16, 2026  
**Dataset Manifest**: `{target_csv.relative_to(REPO_ROOT)}`  
**Total Images Analyzed**: **{len(df):,}**  
**Crops Analyzed**: **Tomato, Potato, Pepper** ({len(df['class_name'].unique())} total classes)  

---

## 1. Dataset Overview & Crop Volume

| Crop | Total Images | Canonical Classes | Train Split | Val Split | Test Split |
| :--- | :---: | :---: | :---: | :---: | :---: |
"""
    for crop in crops:
        st = crop_stats[crop]
        report_md += f"| **{crop}** | **{st['total']:,}** | {st['classes']} | {st['train']:,} | {st['val']:,} | {st['test']:,} |\n"

    report_md += f"""
---

## 2. Crop-Level Imbalance Statistics

| Crop | Min Class Count | Max Class Count | Mean Class Count | Imbalance Ratio (Max / Min) |
| :--- | :---: | :---: | :---: | :---: |
"""
    for crop in crops:
        ist = imbalance_stats[crop]
        report_md += f"| **{crop}** | {ist['min']:,} | {ist['max']:,} | {ist['mean']} | **{ist['ratio']}x** |\n"

    report_md += """
---

## 3. Full Per-Class Distribution & Computed Weights

The class weights are computed per crop using the standard inverse-frequency formula:
$$w_i = \\text{clip}\\left(\\frac{N}{K \\cdot n_i}, 0.1, 10.0\\right)$$

"""
    for crop in crops:
        report_md += f"### {crop} Classes ({crop_stats[crop]['classes']} Classes)\n\n"
        report_md += "| Class Name | Image Count | Crop Percentage | Computed Class Weight |\n"
        report_md += "| :--- | :---: | :---: | :---: |\n"
        counts = per_class_summary[crop]
        N_c = crop_stats[crop]["total"]
        w_c = weights_dict["per_crop"][crop]
        for cls, cnt in counts.items():
            pct = (cnt / N_c) * 100
            report_md += f"| `{cls}` | **{cnt:,}** | {pct:.2f}% | **{w_c[cls]:.4f}** |\n"
        report_md += "\n"

    REPORT_MD_PATH.write_text(report_md, encoding="utf-8")
    print(f"   Saved Markdown Report   : {REPORT_MD_PATH.relative_to(REPO_ROOT)}")

    print("\n=====================================================================")
    print("  CLASS IMBALANCE ANALYSIS COMPLETED SUCCESSFULLY (NO TRAINING RUN)")
    print("=====================================================================\n")

if __name__ == "__main__":
    main()
