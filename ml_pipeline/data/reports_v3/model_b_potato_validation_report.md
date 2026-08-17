# ZARI.ai — Model B Potato Final Validation Report

**Training Date**: August 17, 2026  
**Crop Target**: **Model B Potato** (3 Supervised Classes)  
**Best Epoch**: **Epoch 8** (Best Val Macro F1: **0.9765**)  
**Validation Accuracy**: **97.52%** | **Balanced Acc**: **97.98%**  

---

## 1. Per-Class Validation Breakdown

| Disease Class Label | Label ID | Val Support | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Potato_Early_Blight** | 0 | 275 | 0.9852 | 0.9709 | **0.9780** |
| **Potato_Late_Blight** | 1 | 253 | 0.9684 | 0.9684 | **0.9684** |
| **Potato_Healthy** | 2 | 116 | 0.9667 | 1.0000 | **0.9831** |

---

## 2. Validation Confusion Matrix

```text
[[267   8   0]
 [  4 245   4]
 [  0   0 116]]
```

---

## 3. EDL Epistemic Uncertainty Profile

- **Mean Uncertainty (Correct Predictions)**: `0.0527`
- **Mean Uncertainty (Incorrect Predictions)**: `0.1713`

---

## 4. Per-Epoch History Table

|   epoch | stage               |   train_loss |   val_loss |   train_accuracy |   val_accuracy |   train_macro_f1 |   val_macro_f1 |   mean_val_uncertainty |   learning_rate_backbone |   learning_rate_head |   generalization_gap | diagnostic_state   |
|--------:|:--------------------|-------------:|-----------:|-----------------:|---------------:|-----------------:|---------------:|-----------------------:|-------------------------:|---------------------:|---------------------:|:-------------------|
|       1 | STAGE_1_HEAD_WARMUP |       0.9608 |     0.8055 |           0.8109 |         0.8261 |           0.7968 |         0.8224 |                 0.4374 |                   0      |               0.001  |              -0.0256 | HEALTHY            |
|       2 | STAGE_1_HEAD_WARMUP |       0.6772 |     0.6851 |           0.8837 |         0.8292 |           0.8778 |         0.8241 |                 0.4074 |                   0      |               0.001  |               0.0538 | HEALTHY            |
|       3 | STAGE_1_HEAD_WARMUP |       0.5807 |     0.6367 |           0.8972 |         0.8214 |           0.8937 |         0.8146 |                 0.3844 |                   0      |               0.001  |               0.0791 | HEALTHY            |
|       4 | STAGE_2_FINE_TUNING |       0.228  |     0.2667 |           0.9618 |         0.9115 |           0.9622 |         0.9033 |                 0.1455 |                   0.0001 |               0.001  |               0.0589 | HEALTHY            |
|       5 | STAGE_2_FINE_TUNING |       0.1034 |     0.2072 |           0.9825 |         0.9255 |           0.9836 |         0.9222 |                 0.1131 |                   0.0001 |               0.001  |               0.0614 | HEALTHY            |
|       6 | STAGE_2_FINE_TUNING |       0.0786 |     0.1798 |           0.9884 |         0.9394 |           0.989  |         0.9381 |                 0.0911 |                   0.0001 |               0.001  |               0.0509 | HEALTHY            |
|       7 | STAGE_2_FINE_TUNING |       0.0654 |     0.1279 |           0.9905 |         0.9674 |           0.9912 |         0.9678 |                 0.0728 |                   0.0001 |               0.001  |               0.0234 | HEALTHY            |
|       8 | STAGE_2_FINE_TUNING |       0.0521 |     0.0886 |           0.994  |         0.9752 |           0.9942 |         0.9765 |                 0.0556 |                   0.0001 |               0.001  |               0.0177 | HEALTHY            |
|       9 | STAGE_2_FINE_TUNING |       0.0475 |     0.12   |           0.9952 |         0.9689 |           0.9955 |         0.9699 |                 0.0577 |                   0.0001 |               0.001  |               0.0256 | HEALTHY            |
|      10 | STAGE_2_FINE_TUNING |       0.04   |     0.1537 |           0.9955 |         0.9534 |           0.9958 |         0.9493 |                 0.0583 |                   0.0001 |               0.001  |               0.0465 | HEALTHY            |
|      11 | STAGE_2_FINE_TUNING |       0.0364 |     0.104  |           0.9971 |         0.9767 |           0.9971 |         0.9755 |                 0.0457 |                   5e-05  |               0.0005 |               0.0216 | HEALTHY            |
|      12 | STAGE_2_FINE_TUNING |       0.0319 |     0.1079 |           0.9969 |         0.9658 |           0.9971 |         0.965  |                 0.0457 |                   5e-05  |               0.0005 |               0.0321 | HEALTHY            |
|      13 | STAGE_2_FINE_TUNING |       0.0316 |     0.1117 |           0.9977 |         0.972  |           0.9978 |         0.9724 |                 0.0442 |                   5e-05  |               0.0005 |               0.0254 | EARLY_STOPPING     |