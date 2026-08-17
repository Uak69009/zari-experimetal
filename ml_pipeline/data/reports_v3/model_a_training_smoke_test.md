# ZARI.ai — Model A Crop Router Training Smoke Test Report

**Audit Date**: August 17, 2026  
**Authoritative Manifest**: `ml_pipeline/data/dataset_3crop_final_v4_split.csv`  
**Model Target**: **Model A Crop Router** (Tomato vs. Potato vs. Pepper)  
**Backbone**: Pretrained EfficientNetV2-B2  
**Status**: `SMOKE TEST PASSED — 100% PIPELINE VERIFIED`  

---

## 1. Preflight Verification Checklist

| Acceptance Criteria | Tested Condition | Status |
| :--- | :--- | :---: |
| **Dataset Manifest** | Loaded from `dataset_3crop_final_v4_split.csv` | ✅ PASS |
| **3-Class Mapping** | Tomato=0, Potato=1, Pepper=2 | ✅ PASS |
| **Transform Pipeline** | Dynamic augmentation in Train, Deterministic in Val | ✅ PASS |
| **Batch Shape** | `[64, 3, 256, 256]` RGB Tensor | ✅ PASS |
| **Loss Function** | CrossEntropyLoss with Crop-Level Train Weights | ✅ PASS |
| **AMP & FP16** | PyTorch `torch.amp.autocast('cuda')` & `GradScaler` | ✅ PASS |
| **Backbone Fine-Tuning** | Frozen Head Warmup + Unfrozen Full Backbone | ✅ PASS |
| **Early Stopping Test** | Mock validation history test (Stopped epoch 8) | ✅ PASS |
| **Checkpoint Reload** | State dict saved and reloaded successfully | ✅ PASS |
| **Test Split Protection** | Zero access to test split | ✅ PASS |

---

## 2. Smoke Test Execution Results

- **Stage 1 Loss (Frozen Backbone)**: `1.1181`
- **Stage 2 Loss (Unfrozen Fine-Tuning)**: `0.8638`
- **Validation Loss**: `0.6735`
- **Validation Accuracy**: `75.50%`
- **Validation Macro F1**: `0.7237`
- **Epoch Diagnosis**: `HEALTHY`

---

## FINAL PREFLIGHT VERDICT

```text
SMOKE_TEST_PASS — READY FOR FULL MODEL A TRAINING
```