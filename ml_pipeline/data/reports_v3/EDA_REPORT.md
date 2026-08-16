# Master Dataset V3 — Exploratory Data Analysis (EDA) Report

**Report Date**: August 16, 2026  
**Dataset Path**: `ml_pipeline/data/dataset_final_training_v3.csv`  

---

## 1. Dataset Overview

- **Total Images**: **49,517**
- **Total Crops**: **3** (Tomato, Pepper, Potato)
- **Total Classes**: **32**
- **Train / Val / Test Split Breakdown**:
  - `train`: **39,608** images (79.99%)
  - `val`: **4,953** images (10.00%)
  - `test`: **4,956** images (10.01%)

![Split Distribution](file:///home/hammad/Desktop/project zari - experimental/ml_pipeline/data/reports_v3/plots/eda_split_distribution.png)

---

## 2. Per-Crop Breakdown

| Crop Name | Total Images | Class Count | Percentage |
| :--- | :---: | :---: | :---: |
| **Tomato** | **35,217** | 18 classes | 71.12% |
| **Pepper** | **7,799** | 7 classes | 15.75% |
| **Potato** | **6,501** | 7 classes | 13.13% |

![Crop Volumes](file:///home/hammad/Desktop/project zari - experimental/ml_pipeline/data/reports_v3/plots/eda_crop_volumes.png)

---

## 3. Class Imbalance Analysis

- **Minimum Images per Class**: 2
- **Maximum Images per Class**: 5,433
- **Mean Images per Class**: 1547.4
- **Median Images per Class**: 1287.5

### Flagged Classes (< 50 Images)
- **Tomato_Yellow_Leaf_Curl_Virus**: 48 images (Small Class Flag)
- **Potato_Viral_Leaf_Roll**: 33 images (Small Class Flag)
- **Potato_Bacterial_Soft_Rot**: 7 images (Small Class Flag)
- **Potato_Viral_PVX**: 6 images (Small Class Flag)
- **Potato_Viral_PVY**: 2 images (Small Class Flag)

### Flagged Classes (> 2000 Images)
- **Tomato_Yellow_Curl_Virus**: 5,433 images
- **Tomato_Bacterial_Spot**: 3,942 images
- **Pepper_Bacterial_Spot**: 3,594 images
- **Tomato_Healthy**: 3,585 images
- **Tomato_Septoria**: 3,521 images
- **Tomato_Late_Blight**: 3,476 images
- **Tomato_Early_Blight**: 3,094 images
- **Tomato_Mold**: 2,953 images
- **Potato_Early_Blight**: 2,743 images
- **Potato_Late_Blight**: 2,539 images
- **Tomato_Spider_Mites**: 2,313 images

![Class Distribution](file:///home/hammad/Desktop/project zari - experimental/ml_pipeline/data/reports_v3/plots/eda_class_distribution.png)

---

## 4. Image Quality & Physical Stats

- **Corrupt / Unopenable Files**: **0**
- **Tiny Images (< 50px resolution)**: **0**
- **Average Width x Height**: 575.4 x 564.5 px
- **Average File Size**: 77.9 KB

---

## 5. Sample Grid (2 Images per Class)

![Sample Grid](file:///home/hammad/Desktop/project zari - experimental/ml_pipeline/data/reports_v3/plots/sample_grid.png)
