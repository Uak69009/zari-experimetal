# ZARI.ai — Model B Tomato Final Validation Report

**Training Date**: August 17, 2026  
**Crop Target**: **Model B Tomato** (13 Supervised Classes)  
**Best Epoch**: **Epoch 8** (Best Val Macro F1: **0.9820**)  
**Validation Accuracy**: **98.28%** | **Balanced Acc**: **98.46%**  

---

## 1. Per-Class Validation Breakdown

| Disease Class Label | Label ID | Val Support | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Tomato_Bacterial_Spot** | 0 | 394 | 0.9724 | 0.9822 | **0.9773** |
| **Tomato_Early_Blight** | 1 | 309 | 0.9770 | 0.9612 | **0.9690** |
| **Tomato_Fusarium_Wilt** | 2 | 40 | 0.9302 | 1.0000 | **0.9639** |
| **Tomato_Healthy** | 3 | 355 | 0.9972 | 0.9972 | **0.9972** |
| **Tomato_Late_Blight** | 4 | 343 | 0.9825 | 0.9825 | **0.9825** |
| **Tomato_Leaf_Mold** | 5 | 309 | 0.9708 | 0.9676 | **0.9692** |
| **Tomato_Miner** | 6 | 189 | 0.9947 | 1.0000 | **0.9974** |
| **Tomato_Mosaic_Virus** | 7 | 42 | 1.0000 | 0.9762 | **0.9880** |
| **Tomato_Septoria_Leaf_Spot** | 8 | 358 | 0.9693 | 0.9693 | **0.9693** |
| **Tomato_Spider_Mites** | 9 | 231 | 0.9870 | 0.9870 | **0.9870** |
| **Tomato_Target_Spot** | 10 | 140 | 1.0000 | 0.9857 | **0.9928** |
| **Tomato_Verticillium_Wilt** | 11 | 51 | 0.9623 | 1.0000 | **0.9808** |
| **Tomato_Yellow_Leaf_Curl_Virus** | 12 | 735 | 0.9918 | 0.9905 | **0.9912** |

---

## 2. Validation Confusion Matrix

```text
[[387   1   0   0   0   1   0   0   5   0   0   0   0]
 [  1 297   0   0   4   3   0   0   2   0   0   0   2]
 [  0   0  40   0   0   0   0   0   0   0   0   0   0]
 [  0   0   0 354   0   0   0   0   0   0   0   0   1]
 [  0   1   0   0 337   2   0   0   0   0   0   2   1]
 [  1   0   0   0   2 299   1   0   3   2   0   0   1]
 [  0   0   0   0   0   0 189   0   0   0   0   0   0]
 [  1   0   0   0   0   0   0  41   0   0   0   0   0]
 [  4   3   3   0   0   0   0   0 347   0   0   0   1]
 [  0   0   0   0   0   3   0   0   0 228   0   0   0]
 [  0   1   0   0   0   0   0   0   1   0 138   0   0]
 [  0   0   0   0   0   0   0   0   0   0   0  51   0]
 [  4   1   0   1   0   0   0   0   0   1   0   0 728]]
```

---

## 3. EDL Epistemic Uncertainty Profile

- **Mean Uncertainty (Correct Predictions)**: `0.0794`
- **Mean Uncertainty (Incorrect Predictions)**: `0.3385`

---

## 4. Per-Epoch History Table

|   epoch | stage               |   train_loss |   val_loss |   train_accuracy |   val_accuracy |   train_macro_f1 |   val_macro_f1 |   mean_val_uncertainty |   learning_rate_backbone |   learning_rate_head |   generalization_gap | diagnostic_state   |
|--------:|:--------------------|-------------:|-----------:|-----------------:|---------------:|-----------------:|---------------:|-----------------------:|-------------------------:|---------------------:|---------------------:|:-------------------|
|       1 | STAGE_1_HEAD_WARMUP |       2.0534 |     1.906  |           0.5629 |         0.5981 |           0.5233 |         0.5162 |                 0.502  |                   0      |               0.001  |               0.0071 | HEALTHY            |
|       2 | STAGE_1_HEAD_WARMUP |       1.6649 |     1.8095 |           0.6516 |         0.6207 |           0.6022 |         0.5261 |                 0.4914 |                   0      |               0.001  |               0.0761 | HEALTHY            |
|       3 | STAGE_1_HEAD_WARMUP |       1.5608 |     1.7821 |           0.6825 |         0.633  |           0.6332 |         0.5368 |                 0.4977 |                   0      |               0.001  |               0.0964 | HEALTHY            |
|       4 | STAGE_2_FINE_TUNING |       0.7026 |     0.4907 |           0.8989 |         0.9388 |           0.8768 |         0.9154 |                 0.2127 |                   0.0001 |               0.001  |              -0.0386 | HEALTHY            |
|       5 | STAGE_2_FINE_TUNING |       0.3096 |     0.2419 |           0.9666 |         0.976  |           0.9593 |         0.9686 |                 0.1412 |                   0.0001 |               0.001  |              -0.0093 | HEALTHY            |
|       6 | STAGE_2_FINE_TUNING |       0.24   |     0.1901 |           0.9751 |         0.9797 |           0.97   |         0.973  |                 0.1122 |                   0.0001 |               0.001  |              -0.003  | HEALTHY            |
|       7 | STAGE_2_FINE_TUNING |       0.1952 |     0.1732 |           0.9814 |         0.9823 |           0.9763 |         0.9769 |                 0.0959 |                   0.0001 |               0.001  |              -0.0005 | HEALTHY            |
|       8 | STAGE_2_FINE_TUNING |       0.1698 |     0.1656 |           0.9848 |         0.9828 |           0.9808 |         0.982  |                 0.0839 |                   0.0001 |               0.001  |              -0.0012 | HEALTHY            |
|       9 | STAGE_2_FINE_TUNING |       0.1592 |     0.1641 |           0.9855 |         0.9817 |           0.9826 |         0.9756 |                 0.0743 |                   0.0001 |               0.001  |               0.007  | HEALTHY            |
|      10 | STAGE_2_FINE_TUNING |       0.1463 |     0.1832 |           0.9863 |         0.9826 |           0.9834 |         0.9777 |                 0.0676 |                   0.0001 |               0.001  |               0.0057 | HEALTHY            |
|      11 | STAGE_2_FINE_TUNING |       0.14   |     0.165  |           0.9883 |         0.9817 |           0.9853 |         0.976  |                 0.0607 |                   0.0001 |               0.001  |               0.0093 | HEALTHY            |
|      12 | STAGE_2_FINE_TUNING |       0.1239 |     0.1705 |           0.9892 |         0.9811 |           0.9871 |         0.9749 |                 0.0547 |                   5e-05  |               0.0005 |               0.0123 | HEALTHY            |
|      13 | STAGE_2_FINE_TUNING |       0.1082 |     0.154  |           0.9916 |         0.9837 |           0.9897 |         0.9802 |                 0.0526 |                   5e-05  |               0.0005 |               0.0095 | EARLY_STOPPING     |