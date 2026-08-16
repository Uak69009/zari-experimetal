# ZARI.ai — `new_Dataset` Comprehensive Analysis & Audit Report

**Report Date**: August 16, 2026  
**Source Directory**: `ml_pipeline/data/new_Dataset`  
**Total Candidate Images**: **23,736**  

---

## 1. Summary of Datasets in `new_Dataset`

- **Total Images**: **23,736**
- **Total Crop Species**: **3** (Pepper, Tomato, Potato)
- **Total Diagnostic Classes**: **19**
- **Total Unique Diseases**: **15**
- **Original Non-Augmented Images**: **14,269** images
- **Synthetic Augmented Copies**: **9,467** images

---

## 2. Crop Volume Breakdown

![Crop Volume Breakdown](file:///home/hammad/Desktop/project zari - experimental/ml_pipeline/data/reports_v3/plots/new_dataset_crop_breakdown.png)

| Crop Name | Total Images | Classes Covered | Percentage |
| :--- | :---: | :---: | :---: |
| **Pepper** | **9,283** | 6 classes | 39.11% |
| **Tomato** | **8,030** | 6 classes | 33.83% |
| **Potato** | **6,423** | 7 classes | 27.06% |

---

## 3. Raw Dataset Source Breakdown (Original vs Synthetic)

![Original vs Synthetic Breakdown](file:///home/hammad/Desktop/project zari - experimental/ml_pipeline/data/reports_v3/plots/new_dataset_augmented_vs_raw.png)

| Dataset Folder Name | Total Files | Original Raw | Synthetic Augmented | Environment |
| :--- | :---: | :---: | :---: | :---: |
| **Pepper Bell Leaf Disease** | 9,283 | 9,283 | 0 | FIELD / NATURAL |
| **Tomato Leaf Disease (Pakistan Field)** | 8,030 | 830 | 7,200 | FIELD (Original) |
| **Potato PLD (Central Punjab)** | 4,072 | 4,072 | 0 | FIELD (Original) |
| **Potato Leaf Disease (Bangladesh Field)** | 2,351 | 84 | 2,267 | FIELD (Original) |

---

## 4. Complete Per-Class Distribution (19 Classes)

![Class Distribution](file:///home/hammad/Desktop/project zari - experimental/ml_pipeline/data/reports_v3/plots/new_dataset_class_distribution.png)

| Crop | Class Name | Disease / Condition | Total Images | Original Raw |
| :--- | :--- | :--- | :---: | :---: |
| Pepper | `Pepper_Bacterial_Spot` | Bacterial Spot | **4,901** | 4,901 |
| Potato | `Potato_Late_Blight` | Late Blight | **1,816** | 1,444 |
| Pepper | `Pepper_Cercospora_Leaf_Spot` | Cercospora Leaf Spot | **1,796** | 1,796 |
| Potato | `Potato_Early_Blight` | Early Blight | **1,628** | 1,628 |
| Pepper | `Pepper_Healthy` | Healthy | **1,524** | 1,524 |
| Potato | `Potato_Healthy` | Healthy | **1,412** | 1,036 |
| Tomato | `Tomato_Septoria_Leaf_Spot` | Septoria Leaf Spot | **1,395** | 195 |
| Tomato | `Tomato_Leaf_Mold` | Leaf Mold | **1,386** | 186 |
| Tomato | `Tomato_Healthy` | Healthy | **1,337** | 137 |
| Tomato | `Tomato_Early_Blight` | Early Blight | **1,319** | 119 |
| Tomato | `Tomato_Late_Blight` | Late Blight | **1,313** | 113 |
| Tomato | `Tomato_Yellow_Leaf_Curl_Virus` | Yellow Leaf Curl Virus | **1,280** | 80 |
| Pepper | `Pepper_Nutrition_Deficiency` | Nutrition Deficiency | **444** | 444 |
| Pepper | `Pepper_Leaf_Curl` | Leaf Curl | **423** | 423 |
| Potato | `Potato_Bacterial_Soft_Rot` | Bacterial Soft Rot | **397** | 7 |
| Potato | `Potato_Viral_Leaf_Roll` | Viral Leaf Roll | **394** | 33 |
| Potato | `Potato_Viral_PVY` | Viral PVY | **389** | 2 |
| Potato | `Potato_Viral_PVX` | Viral PVX | **387** | 6 |
| Pepper | `Pepper_Powdery_Mildew` | Powdery Mildew | **195** | 195 |
