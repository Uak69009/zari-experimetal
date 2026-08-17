# ZARI.ai — Final Leaf Segmentation & Severity Architecture Report

**Audit Date**: August 16, 2026  
**Master Manifest**: `ml_pipeline/data/dataset_3crop_final.csv` (**49,805 records**)  
**Scope**: **STRICTLY 3 CROPS** (Tomato, Potato, Bell Pepper)  
**Implementation Policy**: Zero-Shot SAM/SAM2 + Grad-CAM + HSV Heuristic (Zero file modifications, zero manual pixel masks)

---

## 1. Zero-Shot SAM/SAM2 Feasibility Analysis

- **Prompt Strategy**: Bounding Box Prompt centered on the primary leaf region (central 80% ROI) or grid points.
- **Target Leaf Selection**: Combined **Centrality + Largest Mask Area** heuristic.
- **Zero-Shot Boundary**: Generalizes to tomato, potato, and pepper leaves without dataset fine-tuning.

---

## 2. Mask Quality Control Rules

- **ACCEPT**: Mask Area $\in [10\%, 90\%]$ of image, Stability Score $\ge 0.88$, Predicted IoU $\ge 0.85$.
- **REJECT**: Mask Area $< 5\%$ (too small) or $> 95\%$ (background leak), Stability Score $< 0.80$.
- **FALLBACK**: If REJECTED, fallback to **Full Image Classification** with zero pipeline failure.

---

## 3. Disease Region & Severity Estimation Proxy

- **Formulation**: 
  $$\text{Visual Disease Coverage \%} = \frac{\text{Pixels in (SAM Leaf Mask } \cap \text{ Grad-CAM Heatmap } \cap \text{ Lesion Color Filter)}}{\text{Total Pixels in SAM Leaf Mask}} \times 100$$

- **Severity Categories**:
  - **Low**: $< 15\%$ affected leaf area
  - **Medium**: $15\% - 40\%$ affected leaf area
  - **High**: $> 40\%$ affected leaf area

---

## 4. Allowed vs. Forbidden Claims Policy

- ✅ **ALLOWED**: *"Estimated visual disease coverage: 23% (Moderate)"*, *"Relative lesion area proxy: 18%"*.
- ❌ **FORBIDDEN**: *"23% true biological disease severity"*, *"Pixel-perfect disease segmentation"*, *"Pathogen tissue infection rate"*.

---

## FINAL ZARI.ai SEGMENTATION SPECIFICATION

```text
1. Classifier Input     : Original RGB Image (256x256 px)
2. Disease Classification: Model A Crop Router -> Model B EDL Classifier (Disease + Confidence + Uncertainty)
3. SCRC Gate            : Accept prediction if Uncertainty u <= tau, else Reject ('Retake Photo')
4. Leaf Segmentation    : Zero-Shot SAM2 (Central Bounding Box Prompt)
5. Mask Quality Check   : Accept if Area in [10%, 90%] and Stability >= 0.88; else Fallback to Full Image
6. Attention Map        : Grad-CAM on EfficientNetV2-B2 features.7 layer
7. Lesion Intersection  : Mask_Lesion = SAM_Leaf_Mask AND GradCAM_Heatmap (>= 0.40) AND HSV_Necrosis_Filter
8. Severity Calculation : Severity_% = (Count(Mask_Lesion) / Count(SAM_Leaf_Mask)) * 100
9. Severity Label       : Low (<15%), Medium (15-40%), High (>40%)
10. Inference Output    : Disease Name, Model Confidence %, Evidential Uncertainty, Estimated Visual Severity %
```