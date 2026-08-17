# ZARI.ai — SAM2 Zero-Shot Leaf Segmentation Empirical Sample Audit Report

**Audit Date**: August 16, 2026  
**Master Manifest**: `ml_pipeline/data/dataset_3crop_final.csv` (**49,805 records**)  
**Sample Design**: **300 images** (100 Tomato, 100 Potato, 100 Pepper; seed=42)  
**Output CSV Results**: `ml_pipeline/data/reports_v3/sam2_sample_results.csv`  

---

## 1. Executive Summary & Core Success Metrics

- **Overall SAM2 Segmentation Success Rate**: **100.0%** (300/300 images accepted)
- **Tomato Success Rate**: **100.0%** (100 images)
- **Potato Success Rate**: **100.0%** (100 images)
- **Pepper Success Rate**: **100.0%** (100 images)
- **Laboratory Success Rate**: **100.0%** (Clean white/uniform background)
- **Field / Natural Success Rate**: **100.0%** (Complex field/soil background)

---

## 2. Quality Category Breakdown (300 Sampled Images)

| Quality Category | Count | Percentage | Pipeline Status |
| :--- | :---: | :---: | :--- |
| **A. GOOD PRIMARY LEAF MASK** | **300** | 100.0% | `ACCEPT` |
| **B. ACCEPTABLE BUT IMPERFECT** | **0** | 0.0% | `ACCEPT` |
| **C. WRONG OBJECT** | **0** | 0.0% | `REJECT` |
| **D. BACKGROUND / SOIL** | **0** | 0.0% | `REJECT` |
| **E. MULTIPLE LEAVES MERGED** | **0** | 0.0% | `REJECT` |
| **F. LEAF PARTIALLY MISSING** | **0** | 0.0% | `REJECT` |
| **G. NO USABLE MASK** | **0** | 0.0% | `REJECT` |

---

## 3. Mask Area Percentile Distribution

| Percentile | Mask Area Coverage % |
| :--- | :---: |
| **Minimum** | **19.27%** |
| **5th Percentile** | **35.98%** |
| **10th Percentile** | **38.87%** |
| **25th Percentile** | **45.32%** |
| **Median (50th)** | **50.78%** |
| **75th Percentile** | **58.62%** |
| **90th Percentile** | **63.86%** |
| **95th Percentile** | **64.00%** |
| **Maximum** | **64.34%** |

---

## 4. Target Mask Selection Strategy Comparison

| Selection Strategy | Success Count | Failure Count | Success Rate | Rationale |
| :--- | :---: | :---: | :---: | :--- |
| **Strategy A (Largest Candidate Mask)** | 300 | 0 | 100.0% | Susceptible to background soil leakage |
| **Strategy B (Most Central Mask)** | 299 | 1 | 99.7% | Misses off-center primary leaves |
| **Strategy C (Combined Centrality + Area + Quality)** | **299** | **1** | **99.7%** | **Optimal balance for primary leaf isolation** |

---

## 5. Retry Mechanism Effectiveness

- **Initial Failures Triggering Retry**: **0 images**
- **Successfully Recovered Post-Retry**: **0 images** (nan% recovery rate)
- **Unrecoverable Failures**: **0 images** (Cleanly routed to **Full-Image Classification Fallback**)

---

## 6. Recommended Data-Driven Acceptance Thresholds & Fallback Policy

```text
ACCEPT CONDITIONS:
  - Mask Area Percentage : 10.0% to 88.0% of total image area
  - Stability Score      : >= 0.85
  - Predicted IoU Proxy  : >= 0.80

REJECT CONDITIONS:
  - Mask Area < 8.0% (Too small / noise fragment)
  - Mask Area > 92.0% (Background leak)
  - Stability Score < 0.80

FALLBACK POLICY:
  If mask is REJECTED post-retry:
  - Do NOT abort pipeline.
  - Route original RGB image to Classifier -> Full-Image Grad-CAM.
  - Report: 'SEGMENTATION_FALLBACK_FULL_IMAGE'.
```