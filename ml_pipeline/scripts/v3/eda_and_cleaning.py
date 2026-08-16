import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

# Configure plotting style
plt.style.use("seaborn-v0_8-darkgrid" if "seaborn-v0_8-darkgrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["font.size"] = 10

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
DATA_DIR = REPO_ROOT / "ml_pipeline" / "data"
REPORTS_V3_DIR = DATA_DIR / "reports_v3"
PLOTS_DIR = REPORTS_V3_DIR / "plots"

V3_CSV_PATH = DATA_DIR / "dataset_final_training_v3.csv"
CLEAN_CSV_PATH = DATA_DIR / "dataset_final_training_v3_clean.csv"

def main():
    print("=====================================================================")
    print("  ZARI.ai — Master Dataset V3 EDA and Cleaning Pipeline")
    print("=====================================================================\n")

    REPORTS_V3_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    if not V3_CSV_PATH.exists():
        raise FileNotFoundError(f"Missing V3 CSV manifest at {V3_CSV_PATH}")

    df = pd.read_csv(V3_CSV_PATH, low_memory=False)
    initial_len = len(df)

    # -------------------------------------------------------------------------
    # PART 1: EXPLORATORY DATA ANALYSIS (EDA)
    # -------------------------------------------------------------------------
    print("[PART 1] Running Exploratory Data Analysis (EDA)...")

    n_crops = df["crop"].nunique()
    n_classes = df["class_name"].nunique()
    splits_dict = df["split"].value_counts().to_dict()

    print(f"  ✓ Total Images      : {initial_len:,}")
    print(f"  ✓ Total Crops       : {n_crops} ({list(df['crop'].unique())})")
    print(f"  ✓ Total Classes     : {n_classes}")
    print(f"  ✓ Split Counts      : {splits_dict}")

    crop_counts = df["crop"].value_counts()
    class_counts = df["class_name"].value_counts()
    source_counts = df["source_dataset"].value_counts() if "source_dataset" in df.columns else pd.Series()

    # Class Imbalance Analysis
    min_cls_count = class_counts.min()
    max_cls_count = class_counts.max()
    mean_cls_count = class_counts.mean()
    median_cls_count = class_counts.median()

    small_classes = class_counts[class_counts < 50].to_dict()
    large_classes = class_counts[class_counts > 2000].to_dict()

    print(f"  ✓ Class Count Stats : Min={min_cls_count}, Max={max_cls_count}, Mean={mean_cls_count:.1f}, Median={median_cls_count:.1f}")
    print(f"  ✓ Classes < 50 imgs : {len(small_classes)} ({list(small_classes.keys())})")
    print(f"  ✓ Classes > 2000 imgs: {len(large_classes)} ({list(large_classes.keys())})")

    # Image Quality / Physical File Stats
    print("\n  Analyzing image file resolutions and file sizes...")
    widths = []
    heights = []
    file_sizes_kb = []
    corrupt_indices = []
    tiny_indices = []

    for idx, row in df.iterrows():
        img_path = row["image_path"]
        if not os.path.exists(img_path):
            corrupt_indices.append(idx)
            continue
        
        try:
            sz = os.path.getsize(img_path) / 1024.0 # KB
            file_sizes_kb.append(sz)

            with Image.open(img_path) as img:
                w, h = img.size
                widths.append(w)
                heights.append(h)
                if w < 50 or h < 50:
                    tiny_indices.append(idx)
        except Exception:
            corrupt_indices.append(idx)

    # Plot 1: Crop Volumes Bar Chart
    plt.figure(figsize=(9, 5))
    bars = plt.bar(crop_counts.index, crop_counts.values, color=["#2ecc71", "#3498db", "#e74c3c"], edgecolor="black", alpha=0.85)
    plt.title("Dataset V3 — Crop Volume Distribution", fontsize=13, fontweight="bold", pad=15)
    plt.ylabel("Image Count", fontsize=11)
    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., h + 300, f"{int(h):,}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    plt.tight_layout()
    chart_crop_path = PLOTS_DIR / "eda_crop_volumes.png"
    plt.savefig(chart_crop_path, dpi=300)
    plt.close()

    # Plot 2: Class Distribution Bar Chart
    plt.figure(figsize=(14, 7))
    sns.barplot(x=class_counts.values, y=class_counts.index, palette="viridis")
    plt.title(f"Dataset V3 — Class Distribution ({n_classes} Classes)", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Image Count", fontsize=12)
    plt.ylabel("Class Name", fontsize=11)
    plt.tight_layout()
    chart_class_path = PLOTS_DIR / "eda_class_distribution.png"
    plt.savefig(chart_class_path, dpi=300)
    plt.close()

    # Plot 3: Split Distribution Pie Chart
    plt.figure(figsize=(6, 6))
    plt.pie(splits_dict.values(), labels=[f"{k.capitalize()}\n({v:,})" for k, v in splits_dict.items()],
            autopct="%1.1f%%", startangle=140, colors=["#2ecc71", "#3498db", "#e74c3c"], explode=(0.05, 0.05, 0.05),
            textprops={"fontsize": 11, "fontweight": "bold"})
    plt.title("Dataset V3 — Split Distribution", fontsize=13, fontweight="bold", pad=15)
    plt.tight_layout()
    chart_split_path = PLOTS_DIR / "eda_split_distribution.png"
    plt.savefig(chart_split_path, dpi=300)
    plt.close()

    # Plot 4: Sample Grid (2 Images per Class)
    unique_classes = sorted(df["class_name"].unique())
    grid_rows = len(unique_classes)
    fig, axes = plt.subplots(grid_rows, 2, figsize=(6, grid_rows * 2.2))
    
    for i, cname in enumerate(unique_classes):
        c_df = df[df["class_name"] == cname]
        sample_paths = c_df["image_path"].head(2).tolist()
        
        for j in range(2):
            ax = axes[i, j] if grid_rows > 1 else axes[j]
            if j < len(sample_paths) and os.path.exists(sample_paths[j]):
                try:
                    img = Image.open(sample_paths[j]).convert("RGB")
                    ax.imshow(img)
                except Exception:
                    ax.text(0.5, 0.5, "Corrupt", ha="center", va="center")
            else:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center")
            
            if j == 0:
                ax.set_ylabel(cname, rotation=0, labelpad=70, ha="right", va="center", fontsize=8, fontweight="bold")
            ax.set_xticks([])
            ax.set_yticks([])

    plt.suptitle("Dataset V3 — Sample Grid (2 Images per Class)", fontsize=14, fontweight="bold", y=1.002)
    plt.tight_layout()
    chart_grid_path = PLOTS_DIR / "sample_grid.png"
    plt.savefig(chart_grid_path, dpi=300, bbox_inches="tight")
    plt.close()

    # Write EDA_REPORT.md
    eda_report_path = REPORTS_V3_DIR / "EDA_REPORT.md"
    eda_md = f"""# Master Dataset V3 — Exploratory Data Analysis (EDA) Report

**Report Date**: August 16, 2026  
**Dataset Path**: `ml_pipeline/data/dataset_final_training_v3.csv`  

---

## 1. Dataset Overview

- **Total Images**: **{initial_len:,}**
- **Total Crops**: **{n_crops}** ({', '.join(df['crop'].unique())})
- **Total Classes**: **{n_classes}**
- **Train / Val / Test Split Breakdown**:
  - `train`: **{splits_dict.get('train', 0):,}** images ({splits_dict.get('train', 0)/initial_len*100:.2f}%)
  - `val`: **{splits_dict.get('val', 0):,}** images ({splits_dict.get('val', 0)/initial_len*100:.2f}%)
  - `test`: **{splits_dict.get('test', 0):,}** images ({splits_dict.get('test', 0)/initial_len*100:.2f}%)

![Split Distribution](file://{chart_split_path})

---

## 2. Per-Crop Breakdown

| Crop Name | Total Images | Class Count | Percentage |
| :--- | :---: | :---: | :---: |
"""
    for crop in df["crop"].unique():
        c_sub = df[df["crop"] == crop]
        eda_md += f"| **{crop}** | **{len(c_sub):,}** | {c_sub['class_name'].nunique()} classes | {len(c_sub)/initial_len*100:.2f}% |\n"

    eda_md += f"""
![Crop Volumes](file://{chart_crop_path})

---

## 3. Class Imbalance Analysis

- **Minimum Images per Class**: {min_cls_count:,}
- **Maximum Images per Class**: {max_cls_count:,}
- **Mean Images per Class**: {mean_cls_count:.1f}
- **Median Images per Class**: {median_cls_count:.1f}

### Flagged Classes (< 50 Images)
"""
    if small_classes:
        for cname, cnt in small_classes.items():
            eda_md += f"- **{cname}**: {cnt} images (Small Class Flag)\n"
    else:
        eda_md += "- *None (All classes have $\\ge 50$ images)*\n"

    eda_md += "\n### Flagged Classes (> 2000 Images)\n"
    for cname, cnt in large_classes.items():
        eda_md += f"- **{cname}**: {cnt:,} images\n"

    eda_md += f"""
![Class Distribution](file://{chart_class_path})

---

## 4. Image Quality & Physical Stats

- **Corrupt / Unopenable Files**: **{len(corrupt_indices)}**
- **Tiny Images (< 50px resolution)**: **{len(tiny_indices)}**
- **Average Width x Height**: {np.mean(widths):.1f} x {np.mean(heights):.1f} px
- **Average File Size**: {np.mean(file_sizes_kb):.1f} KB

---

## 5. Sample Grid (2 Images per Class)

![Sample Grid](file://{chart_grid_path})
"""
    eda_report_path.write_text(eda_md, encoding="utf-8")
    print(f"  ✓ Saved EDA Report: {eda_report_path.name}")

    # -------------------------------------------------------------------------
    # PART 2: CLEANING
    # -------------------------------------------------------------------------
    print("\n[PART 2] Running Dataset Cleaning Pipeline...")

    # Detect duplicate sha256
    dup_sha256 = df[df.duplicated(subset=["sha256"], keep="first")]
    n_dups = len(dup_sha256)

    # Detect alias issues / trailing space issues
    df["class_clean"] = df["class_name"].astype(str).str.strip()
    alias_map = {
        "Tomato_Curl": "Tomato_Yellow_Leaf_Curl_Virus",
        "Tomato_Yellow_Curl_Virus": "Tomato_Yellow_Leaf_Curl_Virus",
        "Tomato_Mold": "Tomato_Leaf_Mold",
        "Tomato_Septoria": "Tomato_Septoria_Leaf_Spot"
    }

    df["class_clean"] = df["class_clean"].apply(lambda x: alias_map.get(x, x))
    alias_fixes = (df["class_name"] != df["class_clean"]).sum()

    # Apply Cleaning
    clean_df = df.copy()

    # Drop corrupt indices & tiny images
    drop_indices = set(corrupt_indices + tiny_indices)
    clean_df = clean_df.drop(index=list(drop_indices), errors="ignore")

    # Drop duplicate sha256
    clean_df = clean_df.drop_duplicates(subset=["sha256"], keep="first")

    # Remove unknown classes if present
    unknown_mask = clean_df["class_clean"].isin(["Tomato_Unknown", "Pepper_Unknown", "Potato_Unknown"])
    n_unknowns = unknown_mask.sum()
    clean_df = clean_df[~unknown_mask].copy()

    # Apply consolidated class name
    clean_df["class_name"] = clean_df["class_clean"]
    clean_df.drop(columns=["class_clean"], inplace=True)
    clean_df["disease"] = clean_df["class_name"].apply(lambda x: x.replace(x.split("_")[0] + "_", ""))

    final_len = len(clean_df)
    total_removed = initial_len - final_len

    # Save Cleaned CSV
    clean_df.to_csv(CLEAN_CSV_PATH, index=False)
    print(f"  ✓ Cleaned Dataset saved to: {CLEAN_CSV_PATH.name}")

    # Write CLEANING_REPORT.md
    cleaning_report_path = REPORTS_V3_DIR / "CLEANING_REPORT.md"
    cleaning_md = f"""# Master Dataset V3 — Dataset Cleaning Report

**Report Date**: August 16, 2026  
**Cleaned CSV Path**: `ml_pipeline/data/dataset_final_training_v3_clean.csv`  

---

## 1. Summary of Cleaning Actions

| Cleaning Check | Issues Found | Action Taken |
| :--- | :---: | :--- |
| **Corrupt Files** | **{len(corrupt_indices)}** | Removed unopenable images |
| **Tiny Images (<50px)** | **{len(tiny_indices)}** | Removed low-resolution images |
| **Exact SHA256 Duplicates** | **{n_dups}** | Dropped duplicate samples |
| **Unmapped Unknown Classes** | **{n_unknowns}** | Filtered uninformative classes |
| **Taxonomy Alias Inconsistencies** | **{alias_fixes}** | Consolidated into canonical class names |

---

## 2. Before vs After Dataset Volumes

- **Initial Dataset V3 Volume**: **{initial_len:,}** images across {n_classes} classes
- **Total Images Removed / Fixed**: **{total_removed:,}** images
- **Final Cleaned Dataset Volume**: **{final_len:,}** images across **{clean_df['class_name'].nunique()}** classes

### Cleaned Crop Breakdown

| Crop Name | Before Cleaning | After Cleaning | Net Delta |
| :--- | :---: | :---: | :---: |
"""
    for crop in sorted(df["crop"].unique()):
        b_cnt = (df["crop"] == crop).sum()
        a_cnt = (clean_df["crop"] == crop).sum()
        cleaning_md += f"| **{crop}** | {b_cnt:,} | **{a_cnt:,}** | {a_cnt - b_cnt:+} |\n"

    cleaning_md += f"""
---

## 3. Final Cleaned Class Registry ({clean_df['class_name'].nunique()} Classes)

"""
    for crop in sorted(clean_df["crop"].unique()):
        c_sub = clean_df[clean_df["crop"] == crop]
        cleaning_md += f"\n### {crop} Classes ({c_sub['class_name'].nunique()} classes)\n\n"
        cleaning_md += "| Class Name | Image Count | Percentage |\n| :--- | :---: | :---: |\n"
        for cname, cnt in c_sub["class_name"].value_counts().items():
            cleaning_md += f"| `{cname}` | {cnt:,} | {cnt/len(c_sub)*100:.2f}% |\n"

    cleaning_report_path.write_text(cleaning_md, encoding="utf-8")
    print(f"  ✓ Saved Cleaning Report: {cleaning_report_path.name}")

    print("\n=====================================================================")
    print("  EDA & CLEANING COMPLETE — OUTPUT FILES SUMMARY")
    print("=====================================================================")
    print(f"  1. EDA Report       : {eda_report_path.relative_to(REPO_ROOT)}")
    print(f"  2. Cleaning Report  : {cleaning_report_path.relative_to(REPO_ROOT)}")
    print(f"  3. Cleaned Dataset  : {CLEAN_CSV_PATH.relative_to(REPO_ROOT)} ({final_len:,} rows)")
    print(f"  4. Sample Grid Chart: {chart_grid_path.relative_to(REPO_ROOT)}")
    print("=====================================================================\n")

if __name__ == "__main__":
    main()
