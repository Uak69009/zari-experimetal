# ZARI.ai — Model A Crop Router Final Training Report

**Training Date**: August 17, 2026  
**Authoritative Manifest**: `ml_pipeline/data/dataset_3crop_final_v4_split.csv`  
**Model Target**: **Model A Crop Router** (Tomato vs. Potato vs. Pepper)  
**Backbone**: Pretrained EfficientNetV2-B2  
**Best Epoch**: **Epoch 8** (Best Val Macro F1: **0.9923**)  
**Final Status**: `MODEL_A_TRAINING_PASS`  

---

## 1. Key Training Metrics Summary

- **Best Validation Accuracy**: **99.48%**
- **Best Validation Macro F1**: **0.9923**
- **Macro Precision**: `0.9911` | **Macro Recall**: `0.9936`

---

## 2. Per-Crop Validation Performance Breakdown

| Crop Class | Label ID | Validation Support | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Tomato** | 0 | 3,496 | 0.9974 | 0.9954 | **0.9964** |
| **Potato** | 1 | 650 | 0.9832 | 0.9877 | **0.9854** |
| **Pepper** | 2 | 832 | 0.9928 | 0.9976 | **0.9952** |

---

## 3. Validation Confusion Matrix

```text
            Pred Tomato   Pred Potato   Pred Pepper
True Tomato    3480         10            6
True Potato    8            642           0
True Pepper    1            1             830
```

---

## 4. Complete Per-Epoch Training History

|   epoch | stage               |   train_loss |   val_loss |   train_accuracy |   val_accuracy |   train_macro_f1 |   val_macro_f1 |   learning_rate_backbone |   learning_rate_head |   generalization_gap | diagnostic_state   |
|--------:|:--------------------|-------------:|-----------:|-----------------:|---------------:|-----------------:|---------------:|-------------------------:|---------------------:|---------------------:|:-------------------|
|       1 | STAGE_1_HEAD_WARMUP |       0.4772 |     0.7078 |           0.8955 |         0.8303 |           0.8628 |         0.7802 |                  0       |              0.001   |               0.0826 | HEALTHY            |
|       2 | STAGE_1_HEAD_WARMUP |       0.4374 |     0.7214 |           0.9088 |         0.8202 |           0.8802 |         0.7712 |                  0       |              0.001   |               0.1091 | HEALTHY            |
|       3 | STAGE_1_HEAD_WARMUP |       0.4347 |     0.6787 |           0.9101 |         0.8636 |           0.882  |         0.8184 |                  0       |              0.001   |               0.0636 | HEALTHY            |
|       4 | STAGE_2_FINE_TUNING |       0.3256 |     0.454  |           0.9784 |         0.9898 |           0.9692 |         0.9847 |                  0.0001  |              0.001   |              -0.0156 | HEALTHY            |
|       5 | STAGE_2_FINE_TUNING |       0.2872 |     0.4419 |           0.9937 |         0.9906 |           0.9906 |         0.9857 |                  0.0001  |              0.001   |               0.0049 | HEALTHY            |
|       6 | STAGE_2_FINE_TUNING |       0.279  |     0.4365 |           0.9961 |         0.9944 |           0.9942 |         0.9911 |                  0.0001  |              0.001   |               0.0031 | HEALTHY            |
|       7 | STAGE_2_FINE_TUNING |       0.277  |     0.438  |           0.9966 |         0.9942 |           0.995  |         0.9913 |                  0.0001  |              0.001   |               0.0037 | HEALTHY            |
|       8 | STAGE_2_FINE_TUNING |       0.2747 |     0.4382 |           0.9977 |         0.9948 |           0.9967 |         0.9923 |                  0.0001  |              0.001   |               0.0043 | HEALTHY            |
|       9 | STAGE_2_FINE_TUNING |       0.274  |     0.4396 |           0.9977 |         0.9926 |           0.9967 |         0.9889 |                  5e-05   |              0.0005  |               0.0078 | HEALTHY            |
|      10 | STAGE_2_FINE_TUNING |       0.2713 |     0.4399 |           0.9987 |         0.9942 |           0.9981 |         0.9913 |                  5e-05   |              0.0005  |               0.0068 | HEALTHY            |
|      11 | STAGE_2_FINE_TUNING |       0.2701 |     0.4371 |           0.9989 |         0.9948 |           0.9984 |         0.9921 |                  5e-05   |              0.0005  |               0.0063 | HEALTHY            |
|      12 | STAGE_2_FINE_TUNING |       0.27   |     0.4389 |           0.999  |         0.994  |           0.9985 |         0.9909 |                  2.5e-05 |              0.00025 |               0.0076 | HEALTHY            |
|      13 | STAGE_2_FINE_TUNING |       0.2687 |     0.4395 |           0.9994 |         0.9948 |           0.9992 |         0.9921 |                  2.5e-05 |              0.00025 |               0.007  | EARLY_STOPPING     |

---

## FINAL STATUS

```text
MODEL_A_TRAINING_PASS
```