# 📈 ZARI.ai — Master Training & Validation Metrics Report

**Generated Date**: August 17, 2026  
**System Targets**: Tomato (13 classes), Potato (3 classes), Bell Pepper (6 classes)  
**JSON Summary File**: [`ml_pipeline/data/reports_v3/all_models_training_history_summary.json`](file:///home/hammad/Desktop/project zari - experimental/ml_pipeline/data/reports_v3/all_models_training_history_summary.json)

---

## 1. Model A: Crop Router (EfficientNetV2-B2)

- **Task**: 3-Crop Classification (Tomato vs Potato vs Pepper)
- **Plot Artifact**: [`ml_pipeline/reports/figures/01_model_a_crop_router_curves.png`](file:///home/hammad/Desktop/project zari - experimental/ml_pipeline/reports/figures/01_model_a_crop_router_curves.png)

### Epoch-by-Epoch Numerical Table
| Epoch | Stage | Train Loss | Val Loss | Train Acc (%) | Val Acc (%) | Val Macro F1 | Gen Gap |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 01 | STAGE_1_HEAD_WARMUP | 0.4772 | 0.7078 | 89.55% | 83.03% | 0.7802 | `0.0826` |
| 02 | STAGE_1_HEAD_WARMUP | 0.4374 | 0.7214 | 90.88% | 82.02% | 0.7712 | `0.1091` |
| 03 | STAGE_1_HEAD_WARMUP | 0.4347 | 0.6787 | 91.01% | 86.36% | 0.8184 | `0.0636` |
| 04 | STAGE_2_FINE_TUNING | 0.3256 | 0.4540 | 97.84% | 98.98% | 0.9847 | `-0.0156` |
| 05 | STAGE_2_FINE_TUNING | 0.2872 | 0.4419 | 99.37% | 99.06% | 0.9857 | `0.0049` |
| 06 | STAGE_2_FINE_TUNING | 0.2790 | 0.4365 | 99.61% | 99.44% | 0.9911 | `0.0031` |
| 07 | STAGE_2_FINE_TUNING | 0.2770 | 0.4380 | 99.66% | 99.42% | 0.9913 | `0.0037` |
| 08 | STAGE_2_FINE_TUNING | 0.2747 | 0.4382 | 99.77% | 99.48% | 0.9923 | `0.0043` |
| 09 | STAGE_2_FINE_TUNING | 0.2740 | 0.4396 | 99.77% | 99.26% | 0.9889 | `0.0078` |
| 10 | STAGE_2_FINE_TUNING | 0.2713 | 0.4399 | 99.87% | 99.42% | 0.9913 | `0.0068` |
| 11 | STAGE_2_FINE_TUNING | 0.2701 | 0.4371 | 99.89% | 99.48% | 0.9921 | `0.0063` |
| 12 | STAGE_2_FINE_TUNING | 0.2700 | 0.4389 | 99.90% | 99.40% | 0.9909 | `0.0076` |
| 13 | STAGE_2_FINE_TUNING | 0.2687 | 0.4395 | 99.94% | 99.48% | 0.9921 | `0.0070` |

---

## 2. Model B: Crop-Specific EDL Disease Classifiers

- **Loss Function**: Evidential Deep Learning (EDL) Dirichlet Log-Likelihood + Annealed KL Penalty (`kl_penalty = 0.1`)
- **Plot Artifact**: [`ml_pipeline/reports/figures/02_model_b_disease_classifiers_curves.png`](file:///home/hammad/Desktop/project zari - experimental/ml_pipeline/reports/figures/02_model_b_disease_classifiers_curves.png)

### A. Tomato Classifier (13 Classes)
| Epoch | Stage | Train EDL Loss | Val EDL Loss | Train Acc (%) | Val Acc (%) | Val Macro F1 | Mean Val Unc ($u$) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 01 | STAGE_1_HEAD_WARMUP | 2.0534 | 1.9060 | 56.29% | 59.81% | 0.5162 | `0.5020` |
| 02 | STAGE_1_HEAD_WARMUP | 1.6649 | 1.8095 | 65.16% | 62.07% | 0.5261 | `0.4914` |
| 03 | STAGE_1_HEAD_WARMUP | 1.5608 | 1.7821 | 68.25% | 63.30% | 0.5368 | `0.4977` |
| 04 | STAGE_2_FINE_TUNING | 0.7026 | 0.4907 | 89.89% | 93.88% | 0.9154 | `0.2127` |
| 05 | STAGE_2_FINE_TUNING | 0.3096 | 0.2419 | 96.66% | 97.60% | 0.9686 | `0.1412` |
| 06 | STAGE_2_FINE_TUNING | 0.2400 | 0.1901 | 97.51% | 97.97% | 0.9730 | `0.1122` |
| 07 | STAGE_2_FINE_TUNING | 0.1952 | 0.1732 | 98.14% | 98.23% | 0.9769 | `0.0959` |
| 08 | STAGE_2_FINE_TUNING | 0.1698 | 0.1656 | 98.48% | 98.28% | 0.9820 | `0.0839` |
| 09 | STAGE_2_FINE_TUNING | 0.1592 | 0.1641 | 98.55% | 98.17% | 0.9756 | `0.0743` |
| 10 | STAGE_2_FINE_TUNING | 0.1463 | 0.1832 | 98.63% | 98.26% | 0.9777 | `0.0676` |
| 11 | STAGE_2_FINE_TUNING | 0.1400 | 0.1650 | 98.83% | 98.17% | 0.9760 | `0.0607` |
| 12 | STAGE_2_FINE_TUNING | 0.1239 | 0.1705 | 98.92% | 98.11% | 0.9749 | `0.0547` |
| 13 | STAGE_2_FINE_TUNING | 0.1082 | 0.1540 | 99.16% | 98.37% | 0.9802 | `0.0526` |

### B. Potato Classifier (3 Classes)
| Epoch | Stage | Train EDL Loss | Val EDL Loss | Train Acc (%) | Val Acc (%) | Val Macro F1 | Mean Val Unc ($u$) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 01 | STAGE_1_HEAD_WARMUP | 0.9608 | 0.8055 | 81.09% | 82.61% | 0.8224 | `0.4374` |
| 02 | STAGE_1_HEAD_WARMUP | 0.6772 | 0.6851 | 88.37% | 82.92% | 0.8241 | `0.4074` |
| 03 | STAGE_1_HEAD_WARMUP | 0.5807 | 0.6367 | 89.72% | 82.14% | 0.8146 | `0.3844` |
| 04 | STAGE_2_FINE_TUNING | 0.2280 | 0.2667 | 96.18% | 91.15% | 0.9033 | `0.1455` |
| 05 | STAGE_2_FINE_TUNING | 0.1034 | 0.2072 | 98.25% | 92.55% | 0.9222 | `0.1131` |
| 06 | STAGE_2_FINE_TUNING | 0.0786 | 0.1798 | 98.84% | 93.94% | 0.9381 | `0.0911` |
| 07 | STAGE_2_FINE_TUNING | 0.0654 | 0.1279 | 99.05% | 96.74% | 0.9678 | `0.0728` |
| 08 | STAGE_2_FINE_TUNING | 0.0521 | 0.0886 | 99.40% | 97.52% | 0.9765 | `0.0556` |
| 09 | STAGE_2_FINE_TUNING | 0.0475 | 0.1200 | 99.52% | 96.89% | 0.9699 | `0.0577` |
| 10 | STAGE_2_FINE_TUNING | 0.0400 | 0.1537 | 99.55% | 95.34% | 0.9493 | `0.0583` |
| 11 | STAGE_2_FINE_TUNING | 0.0364 | 0.1040 | 99.71% | 97.67% | 0.9755 | `0.0457` |
| 12 | STAGE_2_FINE_TUNING | 0.0319 | 0.1079 | 99.69% | 96.58% | 0.9650 | `0.0457` |
| 13 | STAGE_2_FINE_TUNING | 0.0316 | 0.1117 | 99.77% | 97.20% | 0.9724 | `0.0442` |

### C. Pepper Classifier (6 Classes)
| Epoch | Stage | Train EDL Loss | Val EDL Loss | Train Acc (%) | Val Acc (%) | Val Macro F1 | Mean Val Unc ($u$) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 01 | STAGE_1_HEAD_WARMUP | 1.3479 | 1.0138 | 87.76% | 89.66% | 0.8912 | `0.4999` |
| 02 | STAGE_1_HEAD_WARMUP | 0.8501 | 0.7965 | 94.95% | 91.47% | 0.9146 | `0.4688` |
| 03 | STAGE_1_HEAD_WARMUP | 0.6955 | 0.6858 | 95.12% | 90.99% | 0.9065 | `0.4473` |
| 04 | STAGE_2_FINE_TUNING | 0.2439 | 0.1735 | 98.22% | 95.79% | 0.9654 | `0.1836` |
| 05 | STAGE_2_FINE_TUNING | 0.1055 | 0.1233 | 99.40% | 96.63% | 0.9754 | `0.1255` |
| 06 | STAGE_2_FINE_TUNING | 0.0765 | 0.1127 | 99.64% | 97.00% | 0.9776 | `0.1136` |
| 07 | STAGE_2_FINE_TUNING | 0.0633 | 0.1308 | 99.68% | 95.07% | 0.9695 | `0.1014` |
| 08 | STAGE_2_FINE_TUNING | 0.0575 | 0.0803 | 99.77% | 98.20% | 0.9875 | `0.0795` |
| 09 | STAGE_2_FINE_TUNING | 0.0486 | 0.0627 | 99.85% | 99.04% | 0.9912 | `0.0642` |
| 10 | STAGE_2_FINE_TUNING | 0.0429 | 0.0977 | 99.82% | 96.75% | 0.9791 | `0.0660` |
| 11 | STAGE_2_FINE_TUNING | 0.0380 | 0.0808 | 99.92% | 98.08% | 0.9884 | `0.0560` |
| 12 | STAGE_2_FINE_TUNING | 0.0368 | 0.0668 | 99.94% | 98.44% | 0.9890 | `0.0475` |
| 13 | STAGE_2_FINE_TUNING | 0.0324 | 0.0640 | 99.97% | 98.92% | 0.9934 | `0.0500` |
| 14 | STAGE_2_FINE_TUNING | 0.0307 | 0.0492 | 99.97% | 99.28% | 0.9956 | `0.0468` |
| 15 | STAGE_2_FINE_TUNING | 0.0310 | 0.0682 | 99.97% | 98.20% | 0.9876 | `0.0460` |
| 16 | STAGE_2_FINE_TUNING | 0.0307 | 0.0594 | 99.98% | 98.68% | 0.9919 | `0.0448` |
| 17 | STAGE_2_FINE_TUNING | 0.0265 | 0.0623 | 99.98% | 98.56% | 0.9912 | `0.0453` |
| 18 | STAGE_2_FINE_TUNING | 0.0260 | 0.0630 | 99.95% | 98.80% | 0.9927 | `0.0432` |
| 19 | STAGE_2_FINE_TUNING | 0.0259 | 0.0585 | 100.00% | 98.56% | 0.9897 | `0.0434` |

---

## 3. Swin-Tiny Comparative Evaluation Trajectories

- **Plot Artifact**: [`ml_pipeline/reports/figures/03_swin_vs_efficientnet_f1_trajectories.png`](file:///home/hammad/Desktop/project zari - experimental/ml_pipeline/reports/figures/03_swin_vs_efficientnet_f1_trajectories.png)

### Comparative Validation Macro F1 Matrix
| Crop | EfficientNet Best Val F1 | Swin-Tiny Best Val F1 | F1 Delta | Production Model Choice | Primary Reason |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Tomato** | 0.9820 | **0.9831** | `+0.0011` | **EfficientNetV2-B2** | Grad-CAM 100% Native Compatible. |
| **Potato** | 0.9765 | **0.9882** | `+0.0117` | **EfficientNetV2-B2** | Swin ViT feature tensor $(B,H,W,C)$ breaks Grad-CAM. |
| **Pepper** | 0.9956 | **0.9978** | `+0.0022` | **EfficientNetV2-B2** | Identical early lesion confusion matrix error patterns. |

---

## 4. Knowledge Distillation Trajectories (Swin-Tiny Teacher -> EfficientNet Student)

- **Distillation Params**: Temperature $T=3.0$, Alpha $\alpha=0.7$ (70% Teacher Soft Loss, 30% Hard CE Loss)
- **Plot Artifact**: [`ml_pipeline/reports/figures/04_knowledge_distillation_curves.png`](file:///home/hammad/Desktop/project zari - experimental/ml_pipeline/reports/figures/04_knowledge_distillation_curves.png)

### Epoch-by-Epoch Distillation Table
| Epoch | Total Train Loss | KL Soft Loss ($T=3.0$) | Hard CE Loss | Val Loss | Val Macro F1 | Status |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| 01 | 2.6687 | 3.6224 | 0.4433 | 0.5601 | 0.9740 | Training Pass |
| 02 | 0.3618 | 0.4680 | 0.1141 | 0.4041 | 0.9768 | Best Student Weights |
| 03 | 0.2000 | 0.2464 | 0.0919 | 0.4294 | 0.9750 | Training Pass |
| 04 | 0.1644 | 0.1935 | 0.0965 | 0.4179 | 0.9772 | Training Pass |
| 05 | 0.1183 | 0.1355 | 0.0783 | 0.4772 | 0.9731 | Early Stopping Triggered |
