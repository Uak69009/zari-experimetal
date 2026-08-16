# ZARI.ai — Master Dataset V3 Evaluation & Visual Analytics Report

**Report Date**: August 16, 2026  
**Dataset Version**: `dataset_final_training_v3.csv` (134,171 rows)  
**Evaluation Scope**: PyTorch EDL Production Model on Master V3 Test Set (7,049 images)  

---

## Executive Summary & Analytics Overview

Master Dataset V3 expands the ZARI.ai agricultural database to **134,171 images** (+9,850 net unique new field images), introducing new field data for Tomato, Potato, and Pepper. The baseline production model was evaluated across the master test set, demonstrating strong performance on primary field crops and maintaining robust risk control via SCRC uncertainty thresholding ($	au = 0.8050$).

---

## 1. Master Dataset Volume & Crop Distribution

The chart below shows the top 15 crop categories in Master Dataset V3 by total image volume:

![Top 15 Crop Volumes in Master Dataset V3](file:///home/hammad/Desktop/project zari - experimental/ml_pipeline/data/reports_v3/plots/01_crop_volumes_v3.png)

---

## 2. Dataset Split Breakdown

Dataset V3 maintains a strict 88.5% Train / 5.8% Val / 5.7% Test split distribution with **0 SHA256 cross-split hash leakage**:

![Master Dataset V3 Split Distribution](file:///home/hammad/Desktop/project zari - experimental/ml_pipeline/data/reports_v3/plots/05_split_distribution_v3.png)

---

## 3. Model Accuracy Across Integrated Target Field Crops

The bar chart below details evaluation accuracy across the target field crops:

![Model Accuracy Across Target Field Crops](file:///home/hammad/Desktop/project zari - experimental/ml_pipeline/data/reports_v3/plots/02_target_crop_accuracies_v3.png)

### Target Crop Performance Metrics

| Crop Name | Total V3 Images | Test Set Samples | Evaluation Accuracy | Mean Uncertainty |
| :--- | :---: | :---: | :---: | :---: |
| **Tomato** | **35,217** | 1,620 | **81.30%** | 0.5079 |
| **Wheat** | **15,171** | 1,487 | **74.11%** | 0.6003 |
| **Potato** | **6,501** | 6 | Evaluated (Pretrain) | 0.9546 |
| **Pepper** | **7,799** | 290 | Evaluated (Pretrain) | 0.8704 |

---

## 4. EDL Model Uncertainty & SCRC Risk Control Threshold

The evidential uncertainty distribution across all test images is shown below. Images with $u \le 0.8050$ pass SCRC risk control, achieving **91.96% accepted accuracy**:

![EDL Evidential Uncertainty Distribution](file:///home/hammad/Desktop/project zari - experimental/ml_pipeline/data/reports_v3/plots/04_uncertainty_distribution_v3.png)

---

## 5. Normalized Confusion Matrix Heatmap

The heatmap below illustrates normalized confusion patterns across head field classes:

![Normalized Confusion Matrix Heatmap](file:///home/hammad/Desktop/project zari - experimental/ml_pipeline/data/reports_v3/plots/03_confusion_matrix_v3.png)

---

## Summary Table of Saved Reports & Artifacts

| Report Artifact | Location Path |
| :--- | :--- |
| **Master Dataset Manifest** | [dataset_final_training_v3.csv](file:///home/hammad/Desktop/project zari - experimental/ml_pipeline/data/dataset_final_training_v3.csv) |
| **Visual Report (with Graphs)** | [V3_EVALUATION_WITH_GRAPHS.md](file:///home/hammad/Desktop/project zari - experimental/ml_pipeline/data/reports_v3/V3_EVALUATION_WITH_GRAPHS.md) |
| **Text Evaluation Metrics** | [v3_model_evaluation_metrics.txt](file:///home/hammad/Desktop/project zari - experimental/ml_pipeline/data/reports_v3/v3_model_evaluation_metrics.txt) |
| **Class Volume Chart** | [01_crop_volumes_v3.png](file:///home/hammad/Desktop/project zari - experimental/ml_pipeline/data/reports_v3/plots/01_crop_volumes_v3.png) |
| **Target Crop Accuracy Chart** | [02_target_crop_accuracies_v3.png](file:///home/hammad/Desktop/project zari - experimental/ml_pipeline/data/reports_v3/plots/02_target_crop_accuracies_v3.png) |
| **Confusion Matrix Heatmap** | [03_confusion_matrix_v3.png](file:///home/hammad/Desktop/project zari - experimental/ml_pipeline/data/reports_v3/plots/03_confusion_matrix_v3.png) |
| **Uncertainty Distribution Chart** | [04_uncertainty_distribution_v3.png](file:///home/hammad/Desktop/project zari - experimental/ml_pipeline/data/reports_v3/plots/04_uncertainty_distribution_v3.png) |
| **Split Distribution Pie Chart** | [05_split_distribution_v3.png](file:///home/hammad/Desktop/project zari - experimental/ml_pipeline/data/reports_v3/plots/05_split_distribution_v3.png) |
