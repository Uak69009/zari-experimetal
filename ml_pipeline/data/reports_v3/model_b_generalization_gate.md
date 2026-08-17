# ZARI.ai — Model B Complete Generalization & Robustness Gate Report

**Audit Date**: August 17, 2026  
**Authoritative Manifest**: `ml_pipeline/data/dataset_3crop_final_v4_split.csv` (**49,805 records**)  
**Tomato Checkpoint**: `ml_pipeline/checkpoints/model_b/best_model_b_tomato.pth` (**Epoch 8**)  
**Potato Checkpoint**: `ml_pipeline/checkpoints/model_b/best_model_b_potato.pth` (**Epoch 8**)  
**Pepper Checkpoint**: `ml_pipeline/checkpoints/model_b/best_model_b_pepper.pth` (**Epoch 14**)  
**Overall Status**: `MODEL_B_FROZEN = TRUE — ALL 3 MODELS PASSED`  

---

## 1. Train / Validation / Test Generalization Summary

| Crop Model | Train Acc | Val Acc | Test Acc | Train F1 | Val F1 | Test F1 | Val→Test Gap | ECE |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Tomato** | 99.22% | 98.28% | **98.26%** | 0.9904 | 0.9820 | **0.9783** | `+0.02%` | `0.0688` |
| **Potato** | 99.03% | 97.52% | **96.75%** | 0.9893 | 0.9765 | **0.9718** | `+0.76%` | `0.0243` |
| **Pepper** | 99.44% | 99.28% | **99.40%** | 0.9966 | 0.9956 | **0.9963** | `-0.12%` | `0.0356` |

---

## 2. End-to-End Hierarchical Routing Performance

- **Model A Crop Router Test Accuracy**: `0.00%`
- **Oracle Model B Disease Accuracy (Ground-Truth Routing)**: `98.28%`
- **End-to-End Disease Accuracy (Model A Routing)**: `98.03%`
- **Routing Penalty**: `0.24 %-points` (25 routing errors)

---

## 3. Tier-D Rare Potato EDL Insufficient Evidence Profile

- **Evaluated Rare Potato Images**: `6 images` (PVY, PVX, Soft Rot, Leaf Roll)
- **Mean Epistemic Uncertainty on Rare Potato**: `0.1906`
- **Mean Epistemic Uncertainty on Supported Potato**: `0.0529`
- **Verdict**: Rare Potato images exhibit **3.60x higher epistemic uncertainty**, triggering SCRC selective rejection cleanly.

---

## 4. Final Model Freeze Status

```text
MODEL_B_FROZEN = TRUE
Tomato: FREEZE
Potato: FREEZE
Pepper: FREEZE
```