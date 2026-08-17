# ZARI.ai — Complete Project Workflow & Technical Implementation Progress Report

**Document Date**: August 16, 2026  
**Repository Root**: `project zari - experimental/`  
**Remote GitHub Repo**: [https://github.com/Uak69009/zari-experimetal](https://github.com/Uak69009/zari-experimetal)  
**Current Active Dataset**: [dataset_3crop_final.csv](file:///home/hammad/Desktop/project%20zari%20-%20experimental/ml_pipeline/data/dataset_3crop_final.csv) (**49,805 images**)  

---

## Executive Summary

ZARI.ai is an intelligent agricultural disease diagnosis system for Pakistani farmers. Over the course of development, the project evolved from an initial multi-crop prototype to a production-grade, highly validated, 3-crop master dataset and machine learning pipeline focused on Pakistan's high-priority agricultural crops: **Tomato**, **Potato**, and **Bell Pepper**.

This report documents every technical phase, audit, data pipeline build, MLOps setup, architectural decision, and codebase commit completed in the project.

---

## Phase 1 — Local MLOps Infrastructure & Backend Setup

### 1. MLflow Experiment Tracking
- Integrated MLflow experiment tracking (`zari-phase2`, `zari-model-a`, `zari-model-b`) into model training scripts.
- Launched a local MLflow UI tracking server at `http://127.0.0.1:5000` backed by file store `file:ml_pipeline/mlruns`.
- Backfilled 10 historical training epochs and 9 Selective Classification Risk Control (SCRC) threshold metrics under tag `source: backfilled_from_existing_run`.

### 2. DVC Data & Model Versioning
- Initialized DVC repository and configured local remote storage path at `/home/hammad/dvc_remote_storage`.
- Tracked master dataset manifests (`dataset_3crop_final.csv.dvc`, `dataset_final_training_v2.csv.dvc`) and model checkpoints (`phase2_best.pth`, `phase2_edl_model_v2.pth`). Pushed binary blobs to local storage via `dvc push`.
- Created 2-stage [dvc.yaml](file:///home/hammad/Desktop/project%20zari%20-%20experimental/dvc.yaml) pipeline (`train_phase2` $\rightarrow$ `calibrate`) and verified DAG rendering via `dvc dag`.

### 3. Backend Request Logging
- Updated [backend/main.py](file:///home/hammad/Desktop/project%20zari%20-%20experimental/backend/main.py) with an asynchronous `_log_prediction()` helper logging inference requests, predicted classes, confidence scores, and evidential uncertainties to [backend/monitor_log.jsonl](file:///home/hammad/Desktop/project%20zari%20-%20experimental/backend/monitor_log.jsonl).

---

## Phase 2 — Dataset V2 Analysis & Baseline Evaluation

- Analyzed immutable baseline dataset `dataset_final_training_v2.csv` containing **124,321 rows** across 22 crops and 106 classes.
- Audited Wheat expansion (+1,021 field images to 15,171 total wheat images).
- Evaluated the production PyTorch Evidential Deep Learning (EDL) model (`phase2_edl_model_v2.pth`, EfficientNetV2-S backbone) across the baseline test set, computing accuracy, evidential uncertainty ($u$), and SCRC accepted accuracy under threshold $\tau = 0.8050$.
- Ranked crop image volumes across all 22 crops to ground multi-crop model selection in real data.

---

## Phase 3 — Dataset V3 External Ingestion & Deduplication Pipeline

Four newly downloaded external datasets were ingested under `ml_pipeline/data/new_Dataset` (23,736 total candidate files):
1. **Tomato Pakistan Field Dataset**: 8,030 files
2. **Potato Bangladesh Field Dataset**: 2,351 files
3. **Potato PLD Central Punjab Dataset**: 4,072 files
4. **Bell Pepper Mendeley Dataset**: 9,283 files

### Filtering & Quality Funnel
- **Synthetic Augmentation Quarantine**: Quarantined **10,028 invalid/augmented files** (7,200 Tomato synthetic `aug_` copies and 2,267 Potato `aug_` copies moved to quarantine).
- **Global SHA256 Exact Deduplication**: Dropped **3,858 exact duplicate images**.
- **Net Unique High-Quality Images Integrated**: **+9,850 images**.

---

## Phase 4 — Scope Refinement to 3 Priority Crops

- Refined project scope to focus 100% on Pakistan priority field crops: **Tomato**, **Potato**, and **Bell Pepper**.
- Excluded all 19 non-target crops (Wheat, Apple, Cherry, Corn, Grape, Lokat, etc.) from the active 3-crop master dataset manifest.
- Enforced strict safety rules: **Zero source files or dataset directories were deleted from disk**. All 23,736 candidate files in `new_Dataset` remain 100% intact on disk.

---

## Phase 5 — 7-Phase Construction of `dataset_3crop_final.csv`

1. **Phase 1 (OLD Dataset Audit)**: Loaded `dataset_final_training_v3_clean.csv` (49,232 images), verifying rare Potato classes (`Potato_Viral_PVY`: 2, `Potato_Viral_PVX`: 6, `Potato_Bacterial_Soft_Rot`: 7, `Potato_Viral_Leaf_Roll`: 33) and confirming `Tomato_Curl` (1,893) + `Tomato_Yellow_Curl_Virus` (5,433) = **7,352 images** in `Tomato_Yellow_Leaf_Curl_Virus` against baseline `dataset_final_training_v2.csv`.
2. **Phase 2 (NEW Dataset Audit)**: Scanned `new_Dataset` (23,736 files), confirming that 7,200 Tomato files under `Augmented Dataset/` are synthetic transformed copies of 830 original raw field photos.
3. **Phase 3 (Class Matching)**: Built an explicit alias mapping table (`PepperBell_*`, `Tomato_*`, `Potato_*`). **100% of 23 raw folder categories mapped into 26 canonical classes**.
4. **Phase 4 (Deduplication)**: Dropped **13,696 exact SHA256 duplicate images** matching existing baseline copies, selecting **+573 net unique candidate images**.
5. **Phase 5 (Dataset Assembly)**: Concatenated OLD baseline (49,232 images) + NEW unique images (+573 images) $\rightarrow$ saved to **[dataset_3crop_final.csv](file:///home/hammad/Desktop/project%20zari%20-%20experimental/ml_pipeline/data/dataset_3crop_final.csv)** (**49,805 total images**). Allocated group-atomic SHA256 splits (80% Train / 10% Val / 10% Test) for new rows.
6. **Phase 6 (Validation Audit)**:
   - **Cross-Split SHA256 Leakage**: **`0 Hashes`**
   - **File Path Resolution**: **`100% Resolved`** (0 missing image paths across 49,805 rows)
   - **Duplicate Rows**: **`0`**
   - **Field/Natural Images**: **10,817 real FIELD images** (66.9% Potato Field, 68.0% Pepper Natural)
7. **Phase 7 (Final Summary)**:
   - **Tomato**: 35,015 images (13 classes)
   - **Pepper**: 8,294 images (6 classes)
   - **Potato**: 6,496 images (7 classes)
   - **Splits**: 39,841 Train (80%), 4,978 Val (10%), 4,986 Test (10%)

---

## Phase 6 — Codebase & Git/DVC Synchronization

- Tracked dataset manifest `dataset_3crop_final.csv.dvc` with DVC and pushed binary data to local DVC storage.
- Staged all master scripts, audit scripts, taxonomy configs, analysis reports, and plot figures in Git.
- Committed changes locally (`dbccc9c`) and pushed commit history to GitHub repository: [https://github.com/Uak69009/zari-experimetal.git](https://github.com/Uak69009/zari-experimetal.git).

---

## Phase 7 — Technical Summary & Artifact Inventory

| Category | File Path | Description |
| :--- | :--- | :--- |
| **Master Dataset** | `ml_pipeline/data/dataset_3crop_final.csv` | Final 49,805-row 3-crop dataset manifest |
| **DVC Tracking** | `ml_pipeline/data/dataset_3crop_final.csv.dvc` | DVC pointer file for version control |
| **Build Script** | `ml_pipeline/scripts/v3/build_v3_dataset.py` | Automated dataset build pipeline |
| **EDA & Clean Script**| `ml_pipeline/scripts/v3/eda_and_cleaning.py` | EDA and cleaning analysis pipeline |
| **New Data Analysis**| `ml_pipeline/scripts/v3/analyze_new_dataset.py` | Audit script for `new_Dataset` folder |
| **Evaluation Script**| `ml_pipeline/scripts/v3/evaluate_v3_model.py` | Model evaluation & SCRC calibration script |
| **Plot Script** | `ml_pipeline/scripts/v3/generate_v3_evaluation_plots.py` | Plot generation script |
| **Audit Scripts** | `ml_pipeline/scripts/v3/run_*_audit.py` | Leakage, taxonomy, and dataset volume audit scripts |
| **Taxonomy Config** | `ml_pipeline/config/class_aliases_v3.yaml` | Master taxonomy alias dictionary |
| **Reports** | `ml_pipeline/data/reports_v3/` | Markdown audit reports and 13 plot PNG figures |
| **MLOps Config** | `dvc.yaml`, `backend/main.py` | DVC pipeline DAG and prediction request logger |

---

## Conclusion & Next Steps

Project ZARI.ai now possesses a fully validated, 0-leakage, 100%-resolved 3-crop master dataset ([dataset_3crop_final.csv](file:///home/hammad/Desktop/project%20zari%20-%20experimental/ml_pipeline/data/dataset_3crop_final.csv)) and complete MLOps infrastructure (MLflow + DVC + GitHub).

The codebase and data manifests are fully synchronized and ready for training the two-model hierarchical architecture (**Model A Crop Router** + **Model B Crop Disease Classifiers**).
