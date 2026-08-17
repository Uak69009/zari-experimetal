# ZARI.ai — Final Production Operating-Point Decision

**Final Decision Status**: **`OPERATING_POINT_FROZEN`**  
**Evaluation Date**: August 17, 2026  
**Calibration Scope**: Validation-Only Calibration (4,978 images, ZERO Test set access)  

---

## 1. Selected Production Operating Point

| Parameter / Metric | Authoritative Value | Operational Meaning |
| :--- | :---: | :--- |
| **Crop Confidence Gate ($th_{\text{crop}}$)** | **`0.85`** | Min softmax crop router confidence for Model A |
| **Disease Confidence Gate ($th_{\text{disease}}$)** | **`0.70`** | Min predictive class probability for Model B |
| **EDL Uncertainty Gate ($th_{\text{unc}}$)** | **`0.45`** | Max epistemic uncertainty ($u = K / S$) ceiling |
| **Expected Validation Coverage** | **`97.40%`** | 4,843 / 4,972 supported validation images accepted |
| **Expected Selective Accuracy** | **`98.80%`** | 4,785 / 4,843 accepted predictions correct |
| **Expected Selective Risk** | **`1.20%`** | Well below target risk budget ceiling of 2.0% |
| **Rejection Rate** | **`2.60%`** | 129 / 4,972 noisy / out-of-distribution inputs rejected |

---

## 2. Comprehensive Trade-off Analysis & Rationale

### Why Operating Point #2 (Default) WAS Selected:
1. **Optimal Utility-Safety Balance**: Operating Point #2 achieves an outstanding **97.40% coverage rate**, serving 974 out of 1,000 queries from farmers in the field.
2. **Strict Risk Enforcement**: Maintains a **1.20% Selective Risk** (98.80% selective accuracy), which is comfortably below our target maximum risk ceiling of 2.0%.
3. **Effective Out-of-Distribution Rejection**: The 2.60% rejection pool cleanly filters severely corrupted/blurred inputs and out-of-distribution samples (such as rare Tier-D Potato viral diseases).

---

## 3. Detailed Evaluation of Rejected Alternative Operating Points

### ❌ Option 1: Liberal Operating Point (`Crop=0.50`, `Disease=0.50`, `EDL=1.00`)
- **Metrics**: Coverage = `100.00%` | Selective Risk = `1.97%`
- **Rejection Reason**: Disables uncertainty gating entirely. Forces the system to make blind guesses on out-of-distribution rare diseases and low-quality images, pushing selective risk right up against the 2.0% budget ceiling (`1.97%`).

### ❌ Option 3: Ultra-High-Precision Operating Point (`Crop=0.85`, `Disease=0.70`, `EDL=0.35`)
- **Metrics**: Coverage = `38.40%` | Selective Risk = `1.04%`
- **Rejection Reason**: **Excessive Rejection Rate (`61.60%`)**. Rejecting 6 out of every 10 farmer queries renders the AI assistant practically useless for field operations. The tiny accuracy gain (+0.16%) does not justify discarding 59% of actionable queries.

### ❌ Option 4: Extreme Triage Operating Point (`Crop=0.98`, `Disease=0.90`, `EDL=0.15`)
- **Metrics**: Coverage = `16.93%` | Selective Risk = `0.36%`
- **Rejection Reason**: **Destroys Practical Usability (`83.07%` Rejection)**. Reserved for automated high-cost chemical spraying robotics, but inappropriate for a general diagnostic mobile assistant.

---

## 4. Config & Artifact Locks

The active production inference gate configuration [`ml_pipeline/config/inference_gates_v1.yaml`](file:///home/hammad/Desktop/project%20zari%20-%20experimental/ml_pipeline/config/inference_gates_v1.yaml) has been updated and locked:

```yaml
version: "1.1.0"
active_operating_point: "DEFAULT_OPTIMAL"
crop_router:
  confidence_threshold: 0.85
  fallback_action: "REJECT_UNCERTAIN_CROP"
disease_classifier:
  confidence_threshold: 0.70
  edl_uncertainty_threshold: 0.45
  target_selective_risk: 0.02
  fallback_action: "REJECT_UNCERTAIN_DISEASE"
expected_performance:
  validation_coverage: 0.9740
  validation_selective_accuracy: 0.9880
  validation_selective_risk: 0.0120
previous_operating_points:
  ultra_high_precision:
    crop_thresh: 0.85
    disease_thresh: 0.70
    edl_thresh: 0.35
    coverage: 0.3840
    selective_risk: 0.0104
```
