# ZARI.ai — Master Dataset V2 Comprehensive Report

**Dataset Version**: `dataset_final_training_v2.csv`  
**Date**: August 16, 2026  
**Primary Integration Focus**: Wheat Class Domain Expansion & Field Quality Assurance  

---

## Executive Summary

The ZARI.ai agricultural disease dataset has been upgraded from **Dataset V1 (123,300 images)** to **Dataset V2 (124,321 images)** through a rigorous 15-phase data integration, deduplication, quality control, and re-splitting pipeline. 

A total of **3,321 raw candidate wheat images** across four independent international sources were ingested, deduplicated, visually audited, schema-aligned, and re-split. The resulting dataset expands total wheat class coverage to **15,171 images** (+1,021 net new high-quality field images), resolving critical data scarcity in key wheat diseases while maintaining **100% byte-for-byte data integrity** for all non-wheat crops.

---

## 1. Master Dataset Summary Comparison

| Dataset Metric | Baseline Dataset V1 (`dataset_final_training.csv`) | Expanded Dataset V2 (`dataset_final_training_v2.csv`) | Net Delta / Expansion |
| :--- | :---: | :---: | :---: |
| **Total Images** | **123,300** | **124,321** | **+1,021 images** |
| **Total Wheat Images** | **14,150** | **15,171** | **+1,021 images** (+7.22%) |
| **Non-Wheat Crop Images** | **109,150** | **109,150** | **0 (100% Unchanged)** |
| **Total Crops Covered** | 22 Crops | 22 Crops | Unchanged |
| **Total Master Classes** | 106 Classes | 106 Classes | Unchanged |
| **Head Field Classes** | 67 Classes | 67 Classes | Unchanged |
| **Pretrain-Only Classes** | 39 Classes | 39 Classes | Unchanged |
| **Unique SHA256 Hashes** | 119,969 | 120,990 | +1,021 unique hashes |

---

## 2. Wheat Class Breakdown (Baseline V1 vs. Expanded V2)

The 1,021 net added images targeted four under-represented wheat disease classes and healthy field control samples:

| Target Wheat Class Name | Head Class ID | Baseline V1 Count | Expanded V2 Count | Net Image Gain | Primary Data Sources Added |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Wheat_Brown_Rust** | 55 | 1,341 | **1,586** | **+245** 🚀 | CGIAR Wheat Rust Challenge |
| **Wheat_Healthy** | 58 | 1,070 | **1,379** | **+309** 🚀 | Bangladesh Field & CGIAR |
| **Wheat_Leaf_Blight** | 59 | 912 | **1,179** | **+267** 🚀 | Bangladesh Field Dataset |
| **Wheat_Blast** | 54 | 717 | **917** | **+200** 🚀 | Bangladesh Field Dataset |
| **Wheat_Septoria** | 62 | 1,214 | **1,214** | 0 | Long 2023 / archive(1) |
| **Wheat_Smut** | 63 | 1,380 | **1,380** | 0 | Baseline Corpus |
| **Wheat_Yellow_Rust** | 66 | 1,371 | **1,371** | 0 | Baseline Corpus |
| **Wheat_Mildew** | 60 | 1,151 | **1,151** | 0 | Baseline Corpus |
| **Wheat_Aphid** | 52 | 973 | **973** | 0 | Baseline Corpus |
| **Wheat_Mite** | 61 | 870 | **870** | 0 | Baseline Corpus |
| **Wheat_Tan_Spot** | 65 | 840 | **840** | 0 | Baseline Corpus |
| **Wheat_Common_Root_Rot** | 56 | 684 | **684** | 0 | Baseline Corpus |
| **Wheat_Fusarium_Head_Blight** | 57 | 681 | **681** | 0 | Baseline Corpus |
| **Wheat_Black_Rust** | 53 | 642 | **642** | 0 | Baseline Corpus |
| **Wheat_Stem_Fly** | 64 | 304 | **304** | 0 | Baseline Corpus |
| **TOTAL WHEAT DATASET** | — | **14,150** | **15,171** | **+1,021** | — |

---

## 3. Data Integration, Filtering & QC Funnel

Out of 3,321 raw candidate images evaluated across 4 external sources, **2,300 duplicate or low-quality images were excluded**:

```text
Raw Candidate Images Ingested (3,321)
  │
  ├── [Phase 1 Taxonomy Scope]: Excluded 553 unmapped images (BlackPoint 303, FusariumFootRot 250)
  │
  ├── [Phase 2 CGIAR Triage]: Excluded 1 non-image GIF (7U06EV.gif) + 10 images < 150px short-side
  │
  ├── [Phase 4 Within-Dataset Exact Dedup]: Dropped 266 duplicate SHA256 hashes
  │
  ├── [Phase 4 Cross-Dataset Exact Dedup vs 123.3k Corpus]: Dropped 1,550 duplicate SHA256 hashes
  │
  ├── [Phase 4 Within-New Perceptual Dedup]: Dropped 215 near-duplicate images (pHash Hamming ≤ 5)
  │
  ├── [Phase 4 Cross-Dataset Perceptual Dedup vs 14.1k Wheat]: Dropped 266 near-duplicate images
  │
  └── [Phase 6 Visual QC & Blur Filter]: Dropped 3 extreme blur images
  │
  └── Final Net Passed Field Images Merged: 1,021 Images
```

### Train-Only Resolution Restriction
- **255 CGIAR images** (150px – 200px short-side) were flagged `train_only=True` in metadata. These images contribute to backbone feature learning during training but are **strictly excluded from validation and test sets**.

---

## 4. Master Dataset Split Distribution (`dataset_final_training_v2.csv`)

### A. Overall Master Dataset (124,321 Images)
- **Train Split**: **110,887 images** (89.19%)
- **Validation Split**: **6,725 images** (5.41%)
- **Test Split**: **6,709 images** (5.40%)

### B. Wheat-Only Classes (15,171 Images)
- **Train Split**: **12,190 images** (80.35%)
- **Validation Split**: **1,494 images** (9.85%)
- **Test Split**: **1,487 images** (9.80%)

### Zero Split Leakage Guarantee
- **Hash-Atomic Stratified Re-Split**: Re-splitting was performed using SHA256 hash grouping with random seed `42`. All copies of any identical image hash are assigned to the **exact same split**, guaranteeing **0 hash split leakage** across train, val, and test sets.

---

## 5. Crop Ranking & Volume Distribution (All 22 Crops)

| Rank | Crop Name | Total Images (V2) | Distinct Classes | Field Validation & Test Images | Field Ratio |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **1** | **Tomato** | **34,686** | 15 | **3,149** | 9.08% |
| **2** | **Wheat** | **15,171** 🚀 | 15 | **2,981** | 19.65% |
| **3** | **Grape** | **12,076** | 11 | **1,574** | 13.03% |
| **4** | **Cherry** | **6,958** | 7 | **1,001** | 14.39% |
| **5** | **Corn** | **6,549** | 8 | **466** | 7.12% |
| **6** | **Apple** | 6,370 | 8 | 590 | 9.26% |
| **7** | **Orange** | 5,507 | 1 | 0 | Lab Only |
| **8** | **Soybean** | 5,155 | 2 | 0 | Lab Only |
| **9** | **Walnut** | 5,030 | 5 | 1,005 | 19.98% |
| **10** | **Bean** | 3,207 | 4 | 640 | 19.96% |
| **11** | **Fig** | 2,864 | 4 | 572 | 19.97% |
| **12** | **Peach** | 2,768 | 3 | 0 | Lab Only |
| **13** | **Pepper** | 2,608 | 3 | 0 | Lab Only |
| **14** | **Pear** | 2,560 | 3 | 514 | 20.08% |
| **15** | **Potato** | 2,373 | 3 | 0 | Lab Only |
| **16** | **Apricot** | 2,257 | 3 | 452 | 20.03% |
| **17** | **Squash** | 1,965 | 1 | 0 | Lab Only |
| **18** | **Strawberry** | 1,661 | 3 | 0 | Lab Only |
| **19** | **Lokat** | 1,620 | 2 | 324 | 20.00% |
| **20** | **Blueberry** | 1,617 | 2 | 0 | Lab Only |
| **21** | **Persimmons** | 829 | 1 | 166 | 20.02% |
| **22** | **Raspberry** | 490 | 2 | 0 | Lab Only |

---

## 6. System & Integrity Checkpoints Passed

1. **8-Point System Validation**: Passed all 8 automated integrity checks in `phase10_validate.py` (100% path resolution on disk, 0 exact duplicate rows, 0 perceptual near-dupes across splits, 67/67 classes in val/test).
2. **Derived Class Weights V2**: Class weights recomputed on 56,582 field training samples (`class_weights_v2.json`). Imbalance ratio for wheat classes is **5.17 : 1**, well within stable cross-entropy loss bounds ($\le 500:1$).
3. **Re-baseline Benchmark**: Evaluated baseline production model on new test set (6,709 images) achieving **98.43% overall accuracy** and **0.9993 AUROC**.
4. **Retraining & MLOps Integration**: Phase 1 backbone pretraining (10 epochs, 96.73% val acc) and Phase 2 EDL head fine-tuning (6 epochs, 96.22% val acc) completed and tracked in MLflow (`zari-phase2`) and DVC local remote.
