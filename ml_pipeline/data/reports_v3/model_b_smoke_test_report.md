# ZARI.ai — Model B Complete Smoke Test & EDL Preflight Report

**Audit Date**: August 17, 2026  
**Authoritative Manifest**: `ml_pipeline/data/dataset_3crop_final_v4_split.csv`  
**Frozen Model A Checkpoint**: `ml_pipeline/checkpoints/model_a/best_model_a_efficientnetv2_b2.pth`  
**Status**: `MODEL_B_SMOKE_TEST_PASS — READY FOR FULL TRAINING`  

---

## 1. Crop-Specific Smoke Test Verification Checklist

| Acceptance Criteria | Tested Condition | Status |
| :--- | :--- | :---: |
| **Data Routing** | Ground-truth crop labels used for Model B training | ✅ PASS |
| **Class Mappings** | Tomato (13), Potato (3), Pepper (6) correct | ✅ PASS |
| **Transforms** | Dynamic in Train, Deterministic in Val | ✅ PASS |
| **Loss Function** | EDL Dirichlet NLL Loss + KL Penalty | ✅ PASS |
| **EDL Parametrization** | $\alpha_k = \text{Softplus}(z_k) + 1$, $S = \sum \alpha_k$, $u = K / S$ | ✅ PASS |
| **Differential LRs** | Backbone 1e-4, Head 1e-3 via AdamW | ✅ PASS |
| **AMP & FP16** | `torch.amp.autocast('cuda')` & `GradScaler` | ✅ PASS |
| **Grad-CAM Compatibility** | Target layer `features.7.1` hook verified | ✅ PASS |
| **Tier-D Potato Policy** | Rare classes handled via EDL epistemic uncertainty | ✅ PASS |
| **Test Split Protection** | Zero access to test split | ✅ PASS |

---

## 2. Crop-Specific Smoke Test Performance Results

```json
{
  "Tomato": {
    "num_classes": 13,
    "train_loss": 2.6638,
    "val_loss": 2.4658,
    "train_acc": 0.256,
    "val_acc": 0.355,
    "val_macro_f1": 0.2695,
    "checkpoint": "ml_pipeline/checkpoints/model_b_smoke/smoke_model_b_tomato.pth"
  },
  "Potato": {
    "num_classes": 3,
    "train_loss": 1.1136,
    "val_loss": 0.7796,
    "train_acc": 0.699,
    "val_acc": 0.925,
    "val_macro_f1": 0.9159,
    "checkpoint": "ml_pipeline/checkpoints/model_b_smoke/smoke_model_b_potato.pth"
  },
  "Pepper": {
    "num_classes": 6,
    "train_loss": 1.8927,
    "val_loss": 1.4756,
    "train_acc": 0.692,
    "val_acc": 0.92,
    "val_macro_f1": 0.9388,
    "checkpoint": "ml_pipeline/checkpoints/model_b_smoke/smoke_model_b_pepper.pth"
  }
}
```

---

## FINAL PREFLIGHT VERDICT

```text
MODEL_B_SMOKE_TEST_PASS — READY FOR FULL TRAINING
```