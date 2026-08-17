# ZARI.ai — Class Imbalance & Sampling Strategy Scientific Audit Report

**Audit Date**: August 16, 2026  
**Master Manifest**: `ml_pipeline/data/dataset_3crop_final.csv` (**49,805 records**)  
**Taxonomy Scope**: **26 Canonical Classes** (Tomato, Potato, Bell Pepper)  
**Audit Purpose**: Scientific evaluation of imbalance strategies, extreme minority classes, and loss/sampler formulations before training.

---

## 1. Class Distribution & Imbalance Ratios

- **Total CSV Records**: **49,805**
- **Global Raw Imbalance Ratio (Max / Min)**: **3,676.0x** (`Tomato_Yellow_Leaf_Curl_Virus`: 7,352 vs `Potato_Viral_PVY`: 2)
- **Global Imbalance Ratio (Excl <50 images)**: **37.7x** (`Tomato_Yellow_Leaf_Curl_Virus`: 7,352 vs `Tomato_Fusarium_Wilt`: 407)
- **Global Imbalance Ratio (Excl <100 images)**: **37.7x**

### Per-Crop Imbalance Summary

| Crop Name | Class Count | Total Images | Min Class Size | Max Class Size | Imbalance Ratio (Raw) | Imbalance Ratio (Excl <50) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Tomato** | 13 | 35,015 | 407 (`Fusarium_Wilt`) | 7,352 (`Yellow_Curl`) | **18.1x** | **18.1x** |
| **Pepper** | 6 | 8,294 | 195 (`Powdery_Mildew`) | 4,151 (`Bacterial_Spot`) | **21.3x** | **21.3x** |
| **Potato** | 7 | 6,496 | 2 (`Viral_PVY`) | 2,743 (`Early_Blight`) | **1371.5x** | **2.3x** |

---

## 2. Analysis of the 4 Extreme Minority Classes

The 4 extreme minority Potato classes represent **0.096% of the total dataset** (48 images total):

1. **`Potato_Viral_PVY`** (**2 images**): Bangladesh Field Dataset, 2 Train / 0 Val / 0 Test. IMPOSSIBLE for 3-way split.
2. **`Potato_Viral_PVX`** (**6 images**): Bangladesh Field Dataset, 5 Train / 0 Val / 1 Test. IMPOSSIBLE for validation.
3. **`Potato_Bacterial_Soft_Rot`** (**7 images**): Bangladesh Field Dataset, 4 Train / 3 Val / 0 Test. IMPOSSIBLE.
4. **`Potato_Viral_Leaf_Roll`** (**33 images**): Bangladesh Field Dataset, 30 Train / 2 Val / 1 Test. MARGINAL.

> **Key Rule**: **Synthetic augmentation cannot create new biological variation.** Duplicating 2 images 500 times creates 500 identical visual copies of 2 leaves, leading to 100% memorization and 0% real-world generalization.

---

## 3. Comparison of Weighting Formulas (Sample 10 Classes)

| Class Name | Image Count | Existing Clipped Weight | Raw Inverse-Freq | Sqrt Inverse-Freq | Effective Number |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `Potato_Viral_PVY` | **2** | **10.0000** | 957.7885 | 30.9482 | 0.5000 |
| `Potato_Viral_PVX` | **6** | **5.0212** | 319.2628 | 17.8679 | 0.1667 |
| `Potato_Bacterial_Soft_Rot` | **7** | **4.3039** | 273.6538 | 16.5425 | 0.1429 |
| `Potato_Viral_Leaf_Roll` | **33** | **0.9129** | 58.0478 | 7.6189 | 0.0304 |
| `Pepper_Powdery_Mildew` | **195** | **0.1545** | 9.8235 | 3.1342 | 0.0052 |
| `Tomato_Fusarium_Wilt` | **407** | **0.1000** | 4.7066 | 2.1695 | 0.0025 |
| `Pepper_Leaf_Curl` | **423** | **0.1000** | 4.5286 | 2.1280 | 0.0024 |
| `Tomato_Mosaic_Virus` | **427** | **0.1000** | 4.4861 | 2.1180 | 0.0024 |
| `Pepper_Nutrition_Deficiency` | **444** | **0.1000** | 4.3144 | 2.0771 | 0.0023 |
| `Tomato_Verticillium_Wilt` | **519** | **0.1000** | 3.6909 | 1.9212 | 0.0020 |

---

## 4. FINAL DECISION TABLE

| Class Name | Target Crop | Total Imgs | Tier | Recommended Treatment | Max Multiplier | Loss Weight | Augmentation Policy |
| :--- | :---: | :---: | :---: | :--- | :---: | :---: | :--- |
| `Potato_Viral_PVY` | Potato | 2 | Tier D | Exclude / SCRC Uncertainty | 0.0x | 10.0000 | SCRC Fallback |
| `Potato_Viral_PVX` | Potato | 6 | Tier D | Exclude / SCRC Uncertainty | 0.0x | 5.0212 | SCRC Fallback |
| `Potato_Bacterial_Soft_Rot` | Potato | 7 | Tier D | Exclude / SCRC Uncertainty | 0.0x | 4.3039 | SCRC Fallback |
| `Potato_Viral_Leaf_Roll` | Potato | 33 | Tier D | Exclude / SCRC Uncertainty | 0.0x | 0.9129 | SCRC Fallback |
| `Pepper_Powdery_Mildew` | Pepper | 195 | Tier C | Mild Sampler + Loss Weight | 1.5x | 0.1545 | Moderate Augmentation |
| `Tomato_Fusarium_Wilt` | Tomato | 407 | Tier B | Normal Sampling + Loss Weight | 1.0x | 0.1000 | Moderate Augmentation |
| `Pepper_Leaf_Curl` | Pepper | 423 | Tier B | Normal Sampling + Loss Weight | 1.0x | 0.1000 | Moderate Augmentation |
| `Tomato_Mosaic_Virus` | Tomato | 427 | Tier B | Normal Sampling + Loss Weight | 1.0x | 0.1000 | Moderate Augmentation |
| `Pepper_Nutrition_Deficiency` | Pepper | 444 | Tier B | Normal Sampling + Loss Weight | 1.0x | 0.1000 | Moderate Augmentation |
| `Tomato_Verticillium_Wilt` | Tomato | 519 | Tier B | Normal Sampling + Loss Weight | 1.0x | 0.1000 | Moderate Augmentation |
| `Potato_Healthy` | Potato | 1,171 | Tier A | Normal Sampling | 1.0x | 0.1000 | Baseline Rotation/Flip |
| `Tomato_Target_Spot` | Tomato | 1,404 | Tier A | Normal Sampling | 1.0x | 0.1000 | Baseline Rotation/Flip |
| `Pepper_Healthy` | Pepper | 1,525 | Tier A | Normal Sampling | 1.0x | 0.1000 | Baseline Rotation/Flip |
| `Pepper_Cercospora_Leaf_Spot` | Pepper | 1,556 | Tier A | Normal Sampling | 1.0x | 0.1000 | Baseline Rotation/Flip |
| `Tomato_Miner` | Tomato | 1,897 | Tier A | Normal Sampling | 1.0x | 0.1000 | Baseline Rotation/Flip |
| `Tomato_Spider_Mites` | Tomato | 2,313 | Tier A | Normal Sampling | 1.0x | 0.1000 | Baseline Rotation/Flip |
| `Potato_Late_Blight` | Potato | 2,534 | Tier A | Normal Sampling | 1.0x | 0.1000 | Baseline Rotation/Flip |
| `Potato_Early_Blight` | Potato | 2,743 | Tier A | Normal Sampling | 1.0x | 0.1000 | Baseline Rotation/Flip |
| `Tomato_Leaf_Mold` | Tomato | 3,092 | Tier A | Normal Sampling | 1.0x | 0.1000 | Baseline Rotation/Flip |
| `Tomato_Early_Blight` | Tomato | 3,094 | Tier A | Normal Sampling | 1.0x | 0.1000 | Baseline Rotation/Flip |
| `Tomato_Late_Blight` | Tomato | 3,432 | Tier A | Normal Sampling | 1.0x | 0.1000 | Baseline Rotation/Flip |
| `Tomato_Healthy` | Tomato | 3,550 | Tier A | Normal Sampling | 1.0x | 0.1000 | Baseline Rotation/Flip |
| `Tomato_Septoria_Leaf_Spot` | Tomato | 3,586 | Tier A | Normal Sampling | 1.0x | 0.1000 | Baseline Rotation/Flip |
| `Tomato_Bacterial_Spot` | Tomato | 3,942 | Tier A | Normal Sampling | 1.0x | 0.1000 | Baseline Rotation/Flip |
| `Pepper_Bacterial_Spot` | Pepper | 4,151 | Tier A | Normal Sampling | 1.0x | 0.1000 | Baseline Rotation/Flip |
| `Tomato_Yellow_Leaf_Curl_Virus` | Tomato | 7,352 | Tier A | Normal Sampling | 1.0x | 0.1000 | Baseline Rotation/Flip |

---

## FINAL IMBALANCE STRATEGY

- **Loss Function**: Focal Loss (gamma = 2.0) with Clipped Class Weights (range [0.1, 10.0]).
- **Weight Formula**: Clipped inverse-frequency per crop w_i = clip( (N / (K * n_i)) / mean(N / (K * n_j)), 0.1, 10.0 ).
- **WeightedRandomSampler**: NO (use standard RandomSampler to eliminate minority image memorization).
- **Max Oversampling Cap**: 1.0x (No synthetic duplication of images).
- **Label Smoothing**: epsilon = 0.05.
- **Minimum Viable Class Size for Supervised Loss**: 50 images.

**DO NOT START TRAINING UNTIL THIS STRATEGY HAS BEEN REVIEWED.**