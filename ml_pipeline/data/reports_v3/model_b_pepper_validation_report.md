# ZARI.ai — Model B Pepper Final Validation Report

**Training Date**: August 17, 2026  
**Crop Target**: **Model B Pepper** (6 Supervised Classes)  
**Best Epoch**: **Epoch 14** (Best Val Macro F1: **0.9956**)  
**Validation Accuracy**: **99.28%** | **Balanced Acc**: **99.63%**  

---

## 1. Per-Class Validation Breakdown

| Disease Class Label | Label ID | Val Support | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Pepper_Bacterial_Spot** | 0 | 420 | 0.9952 | 0.9905 | **0.9928** |
| **Pepper_Cercospora_Leaf_Spot** | 1 | 156 | 1.0000 | 0.9872 | **0.9935** |
| **Pepper_Healthy** | 2 | 152 | 0.9744 | 1.0000 | **0.9870** |
| **Pepper_Leaf_Curl** | 3 | 41 | 1.0000 | 1.0000 | **1.0000** |
| **Pepper_Nutrition_Deficiency** | 4 | 43 | 1.0000 | 1.0000 | **1.0000** |
| **Pepper_Powdery_Mildew** | 5 | 20 | 1.0000 | 1.0000 | **1.0000** |

---

## 2. Validation Confusion Matrix

```text
[[416   0   4   0   0   0]
 [  2 154   0   0   0   0]
 [  0   0 152   0   0   0]
 [  0   0   0  41   0   0]
 [  0   0   0   0  43   0]
 [  0   0   0   0   0  20]]
```

---

## 3. EDL Epistemic Uncertainty Profile

- **Mean Uncertainty (Correct Predictions)**: `0.0459`
- **Mean Uncertainty (Incorrect Predictions)**: `0.1748`

---

## 4. Per-Epoch History Table

|   epoch | stage               |   train_loss |   val_loss |   train_accuracy |   val_accuracy |   train_macro_f1 |   val_macro_f1 |   mean_val_uncertainty |   learning_rate_backbone |   learning_rate_head |   generalization_gap | diagnostic_state   |
|--------:|:--------------------|-------------:|-----------:|-----------------:|---------------:|-----------------:|---------------:|-----------------------:|-------------------------:|---------------------:|---------------------:|:-------------------|
|       1 | STAGE_1_HEAD_WARMUP |       1.3479 |     1.0138 |           0.8776 |         0.8966 |           0.8193 |         0.8912 |                 0.4999 |                  0       |              0.001   |              -0.0719 | HEALTHY            |
|       2 | STAGE_1_HEAD_WARMUP |       0.8501 |     0.7965 |           0.9495 |         0.9147 |           0.9364 |         0.9146 |                 0.4688 |                  0       |              0.001   |               0.0217 | HEALTHY            |
|       3 | STAGE_1_HEAD_WARMUP |       0.6955 |     0.6858 |           0.9512 |         0.9099 |           0.9447 |         0.9065 |                 0.4473 |                  0       |              0.001   |               0.0382 | HEALTHY            |
|       4 | STAGE_2_FINE_TUNING |       0.2439 |     0.1735 |           0.9822 |         0.9579 |           0.9793 |         0.9654 |                 0.1836 |                  0.0001  |              0.001   |               0.0139 | HEALTHY            |
|       5 | STAGE_2_FINE_TUNING |       0.1055 |     0.1233 |           0.994  |         0.9663 |           0.9917 |         0.9754 |                 0.1255 |                  0.0001  |              0.001   |               0.0163 | HEALTHY            |
|       6 | STAGE_2_FINE_TUNING |       0.0765 |     0.1127 |           0.9964 |         0.97   |           0.9965 |         0.9776 |                 0.1136 |                  0.0001  |              0.001   |               0.0189 | HEALTHY            |
|       7 | STAGE_2_FINE_TUNING |       0.0633 |     0.1308 |           0.9968 |         0.9507 |           0.9973 |         0.9695 |                 0.1014 |                  0.0001  |              0.001   |               0.0278 | HEALTHY            |
|       8 | STAGE_2_FINE_TUNING |       0.0575 |     0.0803 |           0.9977 |         0.982  |           0.9973 |         0.9875 |                 0.0795 |                  0.0001  |              0.001   |               0.0098 | HEALTHY            |
|       9 | STAGE_2_FINE_TUNING |       0.0486 |     0.0627 |           0.9985 |         0.9904 |           0.9989 |         0.9912 |                 0.0642 |                  0.0001  |              0.001   |               0.0077 | HEALTHY            |
|      10 | STAGE_2_FINE_TUNING |       0.0429 |     0.0977 |           0.9982 |         0.9675 |           0.9989 |         0.9791 |                 0.066  |                  0.0001  |              0.001   |               0.0198 | HEALTHY            |
|      11 | STAGE_2_FINE_TUNING |       0.038  |     0.0808 |           0.9992 |         0.9808 |           0.9995 |         0.9884 |                 0.056  |                  0.0001  |              0.001   |               0.0112 | HEALTHY            |
|      12 | STAGE_2_FINE_TUNING |       0.0368 |     0.0668 |           0.9994 |         0.9844 |           0.9996 |         0.989  |                 0.0475 |                  5e-05   |              0.0005  |               0.0106 | HEALTHY            |
|      13 | STAGE_2_FINE_TUNING |       0.0324 |     0.064  |           0.9997 |         0.9892 |           0.9996 |         0.9934 |                 0.05   |                  5e-05   |              0.0005  |               0.0063 | HEALTHY            |
|      14 | STAGE_2_FINE_TUNING |       0.0307 |     0.0492 |           0.9997 |         0.9928 |           0.9998 |         0.9956 |                 0.0468 |                  5e-05   |              0.0005  |               0.0042 | HEALTHY            |
|      15 | STAGE_2_FINE_TUNING |       0.031  |     0.0682 |           0.9997 |         0.982  |           0.9991 |         0.9876 |                 0.046  |                  5e-05   |              0.0005  |               0.0116 | HEALTHY            |
|      16 | STAGE_2_FINE_TUNING |       0.0307 |     0.0594 |           0.9998 |         0.9868 |           0.9999 |         0.9919 |                 0.0448 |                  5e-05   |              0.0005  |               0.008  | HEALTHY            |
|      17 | STAGE_2_FINE_TUNING |       0.0265 |     0.0623 |           0.9998 |         0.9856 |           0.9999 |         0.9912 |                 0.0453 |                  2.5e-05 |              0.00025 |               0.0087 | HEALTHY            |
|      18 | STAGE_2_FINE_TUNING |       0.026  |     0.063  |           0.9995 |         0.988  |           0.9997 |         0.9927 |                 0.0432 |                  2.5e-05 |              0.00025 |               0.0071 | HEALTHY            |
|      19 | STAGE_2_FINE_TUNING |       0.0259 |     0.0585 |           1      |         0.9856 |           1      |         0.9897 |                 0.0434 |                  2.5e-05 |              0.00025 |               0.0103 | EARLY_STOPPING     |