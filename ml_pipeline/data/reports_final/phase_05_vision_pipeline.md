# Phase 05 — Frozen Calibrated Vision Pipeline Report

Hierarchical Inference Control Flow:
1. Image Input $ightarrow$ Deterministic Preprocessing (256x256 RGB ImageNet normalized)
2. Model A Crop Router $ightarrow$ Softmax Crop Confidence
   - If Crop Confidence $< 0.85 ightarrow$ Rejects with `REJECT_UNCERTAIN_CROP`
3. Model B Crop-Specific Disease Classifier $ightarrow$ Dirichlet Evidence $lpha_k$, Probabilities $p_k$, Epistemic Uncertainty $u = K / S$
   - If Disease Confidence $< 0.70$ OR Epistemic Uncertainty $> 0.35 ightarrow$ Rejects with `REJECT_UNCERTAIN_DISEASE` / `INSUFFICIENT_EVIDENCE`
4. If Accepted $ightarrow$ Passes to downstream SAM2 post-classification segmentation & Grad-CAM analysis.

**Status**: `VISION_PIPELINE_PASS`
