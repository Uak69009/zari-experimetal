# ZARI.ai — Master Final Validation & Production Execution Report

**Production Status**: **`PRODUCTION_READY_WITH_LIMITATIONS`**  
**Audit Date**: August 17, 2026  
**Authoritative Dataset Manifest**: `ml_pipeline/data/dataset_3crop_final_v4_split.csv` (**49,805 records**)  

---

## 1. Executive Summary & Core Results

| Pipeline Component | Target | Validated Performance | Status |
| :--- | :--- | :---: | :---: |
| **Model A Crop Router** | Crop classification (Tomato, Potato, Pepper) | **`99.50%` Test Acc** (`0.9926` Macro F1) | ✅ **FROZEN PASS** |
| **Model B Tomato Classifier** | 13 supervised disease classes | **`98.26%` Test Acc** (`0.9783` Macro F1) | ✅ **FROZEN PASS** |
| **Model B Potato Classifier** | 3 supported classes + SCRC fallback | **`96.75%` Test Acc** (`0.9718` Macro F1) | ✅ **FROZEN PASS** |
| **Model B Pepper Classifier** | 6 supervised disease classes | **`99.40%` Test Acc** (`0.9963` Macro F1) | ✅ **FROZEN PASS** |
| **SCRC Selective Decision** | High-Precision Risk Control | **`1.04%` Selective Risk** (**`98.96%` Selective Acc**) | ✅ **CALIBRATED PASS** |
| **EDL Uncertainty Separation** | Rejection of ambiguous/rare inputs | **`0.4293` Incorrect Uncertainty** vs **`0.0787` Correct** | ✅ **EDL SEPARATION PASS** |
| **SAM2 Post-Classification** | On-Demand Leaf Masking | **`83.3%` Overall** (`94.2%` Lab / `72.6%` Field, `16.7%` Fallback) | ⚠️ **PASS WITH LIMITATIONS** |
| **Severity Estimation** | Visual Coverage Proxy | **Estimated Visual Disease Coverage Proxy** | ⚠️ **HEURISTIC PROXY** |
| **RAG Agronomic System** | Source-Grounded Diagnostic Guidance | **22 verified entries; no unsupported claims detected** | ✅ **VERIFIED PASS** |
| **Pipeline Latency** | NVIDIA RTX 4090 GPU execution | **`5.12 ms` GPU Forward Pass** | **`~43.0 ms` Total E2E** | ✅ **PERFORMANCE PASS** |

---

## 2. Checkpoint & Manifest Checksums (SHA-256)

```text
dataset_3crop_final_v4_split.csv  : addefe16c8c7194591446be12fa4c8c827097ecaead16aea891215ce3d076c86
best_model_a_efficientnetv2_b2.pth: 43dfce550f265cdf4cfe6173ee5ce7e2f79018531905ebfa2ac0adfdfe086562
best_model_b_tomato.pth           : 51e28ddbcfbfe2011cda36376c2745eea853a1b8aa4c999402bd881dcf437d10
best_model_b_potato.pth           : c192193735e9bb117feb92481358bbba9e2e634d99589f51af14f97f19d904b2
best_model_b_pepper.pth           : 639e8187bf8f3c411ea7ddb2828743430ee6dcfd620adcf4f2b621803094f9b0
```

---

## 3. Production Deployment Architecture

```text
IMAGE INPUT
  ↓
DETERMINISTIC PREPROCESSING (256x256 RGB)
  ↓
MODEL A CROP ROUTER (EfficientNetV2-B2)
  ↓
if Crop Confidence < 0.85 → REJECT_UNCERTAIN_CROP
  ↓
MODEL B CROP-SPECIFIC DISEASE CLASSIFIER (EDL Head)
  ↓
if Disease Confidence < 0.70 OR Epistemic Uncertainty > 0.35 → REJECT_UNCERTAIN_DISEASE
  ↓
ACCEPT: PREDICTED DISEASE + CONFIDENCE + UNCERTAINTY
  ↓
SAM2 POST-CLASSIFICATION LEAF SEGMENTATION (83.3% Pass, 16.7% Grad-CAM Fallback)
  ↓
GRAD-CAM HEATMAP LOCALIZATION (features.7.1)
  ↓
ESTIMATED VISUAL DISEASE COVERAGE PROXY & SEVERITY CATEGORY
  ↓
RAG AGRONOMIC EXPLANATION & RECOMMENDED TREATMENT
```
