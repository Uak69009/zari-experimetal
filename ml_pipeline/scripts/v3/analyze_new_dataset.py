import os
import sys
import shutil
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
NEW_DATASET_DIR = DATA_DIR / "new_Dataset"
REPORTS_V3_DIR = DATA_DIR / "reports_v3"
PLOTS_DIR = REPORTS_V3_DIR / "plots"
REPORT_MD_PATH = DATA_DIR / "new_dataset_analysis_report.md"

def main():
    print("=====================================================================")
    print("  ZARI.ai — New Dataset Comprehensive Analysis & Old Data Cleanup")
    print("=====================================================================\n")

    REPORTS_V3_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    if not NEW_DATASET_DIR.exists():
        raise FileNotFoundError(f"new_Dataset directory missing at {NEW_DATASET_DIR}")

    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

    # 1. Scan new_Dataset recursively
    records = []
    for root, dirs, files in os.walk(NEW_DATASET_DIR):
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext in valid_exts:
                fpath = os.path.join(root, f)
                rel_path = os.path.relpath(fpath, NEW_DATASET_DIR)
                parts = rel_path.split(os.sep)
                top_folder = parts[0]
                
                # Identify crop & raw vs augmented status
                crop = "Unknown"
                if "Pepper" in top_folder:
                    crop = "Pepper"
                elif "Tomato" in top_folder:
                    crop = "Tomato"
                elif "Potato" in top_folder or "pld" in top_folder.lower():
                    crop = "Potato"

                is_augmented = f.lower().startswith("aug_") or "Augmented Dataset" in rel_path
                class_folder = parts[-2] if len(parts) >= 2 else "root"
                
                records.append({
                    "image_path": fpath,
                    "filename": f,
                    "top_dataset": top_folder,
                    "crop": crop,
                    "raw_folder": class_folder,
                    "is_augmented": is_augmented
                })

    df_new = pd.DataFrame(records)
    total_imgs = len(df_new)
    print(f"✓ Total Images Scanned in new_Dataset: {total_imgs:,}")

    # Map raw folder names to canonical class/disease names
    def map_canonical(row):
        rf = row["raw_folder"]
        c = row["crop"]
        rf_lower = rf.lower()

        if c == "Pepper":
            if "bacterial" in rf_lower: return "Pepper_Bacterial_Spot", "Bacterial Spot"
            elif "cercospora" in rf_lower: return "Pepper_Cercospora_Leaf_Spot", "Cercospora Leaf Spot"
            elif "healthy" in rf_lower: return "Pepper_Healthy", "Healthy"
            elif "curl" in rf_lower: return "Pepper_Leaf_Curl", "Leaf Curl"
            elif "nutrition" in rf_lower: return "Pepper_Nutrition_Deficiency", "Nutrition Deficiency"
            elif "powdery" in rf_lower or "mildew" in rf_lower: return "Pepper_Powdery_Mildew", "Powdery Mildew"
        elif c == "Potato":
            if "early" in rf_lower: return "Potato_Early_Blight", "Early Blight"
            elif "late" in rf_lower or "fungal" in rf_lower: return "Potato_Late_Blight", "Late Blight"
            elif "healthy" in rf_lower: return "Potato_Healthy", "Healthy"
            elif "soft rot" in rf_lower or "bacterial" in rf_lower: return "Potato_Bacterial_Soft_Rot", "Bacterial Soft Rot"
            elif "leaf roll" in rf_lower or "viral leaf roll" in rf_lower: return "Potato_Viral_Leaf_Roll", "Viral Leaf Roll"
            elif "pvx" in rf_lower: return "Potato_Viral_PVX", "Viral PVX"
            elif "pvy" in rf_lower: return "Potato_Viral_PVY", "Viral PVY"
        elif c == "Tomato":
            if "early" in rf_lower: return "Tomato_Early_Blight", "Early Blight"
            elif "late" in rf_lower: return "Tomato_Late_Blight", "Late Blight"
            elif "healthy" in rf_lower: return "Tomato_Healthy", "Healthy"
            elif "yellow" in rf_lower or "curl" in rf_lower: return "Tomato_Yellow_Leaf_Curl_Virus", "Yellow Leaf Curl Virus"
            elif "mold" in rf_lower: return "Tomato_Leaf_Mold", "Leaf Mold"
            elif "septora" in rf_lower or "septoria" in rf_lower: return "Tomato_Septoria_Leaf_Spot", "Septoria Leaf Spot"

        return f"{c}_{rf}", rf

    mapped = df_new.apply(map_canonical, axis=1)
    df_new["class_name"] = [m[0] for m in mapped]
    df_new["disease"] = [m[1] for m in mapped]

    crop_counts = df_new["crop"].value_counts()
    class_counts = df_new["class_name"].value_counts()
    disease_counts = df_new["disease"].value_counts()
    aug_counts = df_new["is_augmented"].value_counts()

    print(f"✓ Total Crops      : {len(crop_counts)} ({list(crop_counts.index)})")
    print(f"✓ Total Classes    : {len(class_counts)}")
    print(f"✓ Total Diseases   : {len(disease_counts)}")
    print(f"✓ Original Raw     : {(~df_new['is_augmented']).sum():,} images")
    print(f"✓ Synthetic Aug    : {df_new['is_augmented'].sum():,} images (quarantined/filtered in production pipeline)")

    # Plot 1: Crop Breakdown Chart
    plt.figure(figsize=(9, 5))
    bars = plt.bar(crop_counts.index, crop_counts.values, color=["#2ecc71", "#e74c3c", "#3498db"], edgecolor="black", alpha=0.85)
    plt.title("new_Dataset — Crop Volume Breakdown", fontsize=14, fontweight="bold", pad=15)
    plt.ylabel("Image Count", fontsize=12)
    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., h + 200, f"{int(h):,}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    plt.tight_layout()
    plot1_path = PLOTS_DIR / "new_dataset_crop_breakdown.png"
    plt.savefig(plot1_path, dpi=300)
    plt.close()

    # Plot 2: Class Distribution Chart
    plt.figure(figsize=(14, 8))
    sns.barplot(x=class_counts.values, y=class_counts.index, palette="viridis")
    plt.title(f"new_Dataset — Raw Class Distribution ({len(class_counts)} Classes)", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Image Count", fontsize=12)
    plt.tight_layout()
    plot2_path = PLOTS_DIR / "new_dataset_class_distribution.png"
    plt.savefig(plot2_path, dpi=300)
    plt.close()

    # Plot 3: Raw vs Synthetic Augmented Stacked Bar Chart
    dataset_summary = df_new.groupby(["top_dataset", "is_augmented"]).size().unstack(fill_value=0)
    dataset_summary.columns = ["Original Raw", "Synthetic Augmented"]
    dataset_summary.plot(kind="barh", stacked=True, color=["#2ecc71", "#e74c3c"], figsize=(11, 5), edgecolor="black", alpha=0.85)
    plt.title("new_Dataset — Original Raw vs Synthetic Augmented Breakdown", fontsize=13, fontweight="bold", pad=15)
    plt.xlabel("Image Count", fontsize=11)
    plt.ylabel("Dataset Folder", fontsize=11)
    plt.tight_layout()
    plot3_path = PLOTS_DIR / "new_dataset_augmented_vs_raw.png"
    plt.savefig(plot3_path, dpi=300)
    plt.close()

    # Generate Markdown Report
    report_md = f"""# ZARI.ai — `new_Dataset` Comprehensive Analysis & Audit Report

**Report Date**: August 16, 2026  
**Source Directory**: `ml_pipeline/data/new_Dataset`  
**Total Candidate Images**: **{total_imgs:,}**  

---

## 1. Summary of Datasets in `new_Dataset`

- **Total Images**: **{total_imgs:,}**
- **Total Crop Species**: **{len(crop_counts)}** ({', '.join(crop_counts.index)})
- **Total Diagnostic Classes**: **{len(class_counts)}**
- **Total Unique Diseases**: **{len(disease_counts)}**
- **Original Non-Augmented Images**: **{(~df_new['is_augmented']).sum():,}** images
- **Synthetic Augmented Copies**: **{df_new['is_augmented'].sum():,}** images

---

## 2. Crop Volume Breakdown

![Crop Volume Breakdown](file://{plot1_path})

| Crop Name | Total Images | Classes Covered | Percentage |
| :--- | :---: | :---: | :---: |
"""
    for crop, cnt in crop_counts.items():
        n_c = df_new[df_new["crop"] == crop]["class_name"].nunique()
        report_md += f"| **{crop}** | **{cnt:,}** | {n_c} classes | {cnt/total_imgs*100:.2f}% |\n"

    report_md += f"""
---

## 3. Raw Dataset Source Breakdown (Original vs Synthetic)

![Original vs Synthetic Breakdown](file://{plot3_path})

| Dataset Folder Name | Total Files | Original Raw | Synthetic Augmented | Environment |
| :--- | :---: | :---: | :---: | :---: |
| **Pepper Bell Leaf Disease** | 9,283 | 9,283 | 0 | FIELD / NATURAL |
| **Tomato Leaf Disease (Pakistan Field)** | 8,030 | 830 | 7,200 | FIELD (Original) |
| **Potato PLD (Central Punjab)** | 4,072 | 4,072 | 0 | FIELD (Original) |
| **Potato Leaf Disease (Bangladesh Field)** | 2,351 | 84 | 2,267 | FIELD (Original) |

---

## 4. Complete Per-Class Distribution ({len(class_counts)} Classes)

![Class Distribution](file://{plot2_path})

| Crop | Class Name | Disease / Condition | Total Images | Original Raw |
| :--- | :--- | :--- | :---: | :---: |
"""
    for cname, cnt in class_counts.items():
        c_df = df_new[df_new["class_name"] == cname]
        crop = c_df["crop"].iloc[0]
        dis = c_df["disease"].iloc[0]
        orig_cnt = (~c_df["is_augmented"]).sum()
        report_md += f"| {crop} | `{cname}` | {dis} | **{cnt:,}** | {orig_cnt:,} |\n"

    REPORT_MD_PATH.write_text(report_md, encoding="utf-8")
    print(f"\n✓ Saved Comprehensive Report: {REPORT_MD_PATH.name}")

    # -------------------------------------------------------------------------
    # PART 2: CLEAN UP OLD UNUSED DATA
    # -------------------------------------------------------------------------
    print("\n[PART 2] Cleaning up old unused data folders...")

    old_dirs_to_delete = [
        DATA_DIR / "Disease Dataset of Wheat Original, Augmented, and Balanced for Deep Learning",
        DATA_DIR / "Long 2023 Plant Path 999 photos",
        DATA_DIR / "archive(1)",
        DATA_DIR / "cigar crop disease",
        DATA_DIR / "raw_external"
    ]

    deleted_dirs = []
    reclaimed_bytes = 0

    for odir in old_dirs_to_delete:
        if odir.exists() and odir.is_dir():
            try:
                # Calculate size
                for root, dirs, files in os.walk(odir):
                    for f in files:
                        reclaimed_bytes += os.path.getsize(os.path.join(root, f))
                shutil.rmtree(odir)
                deleted_dirs.append(odir.name)
                print(f"  ✓ Deleted old unused folder: {odir.name}")
            except Exception as e:
                print(f"  Error deleting {odir.name}: {e}")

    reclaimed_mb = reclaimed_bytes / (1024 * 1024)
    print(f"\n=====================================================================")
    print("  NEW DATASET ANALYSIS & CLEANUP SUMMARY")
    print("=====================================================================")
    print(f"  Total Images Analyzed in new_Dataset: {total_imgs:,}")
    print(f"  Total Classes / Diseases             : {len(class_counts)} classes across 3 crops")
    print(f"  Old Unused Folders Deleted           : {len(deleted_dirs)} directories")
    print(f"  Storage Space Reclaimed              : {reclaimed_mb:.2f} MB")
    print(f"  Saved Analysis Report                : {REPORT_MD_PATH.name}")
    print("=====================================================================\n")

if __name__ == "__main__":
    main()
