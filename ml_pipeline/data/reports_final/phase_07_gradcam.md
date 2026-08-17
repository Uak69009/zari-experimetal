# Phase 07 — Grad-CAM Localization Report

- **Target Backbone Layer**: `features.7.1` on EfficientNetV2-B2.
- **Mask Intersection**: Grad-CAM heatmap $\cap$ SAM2 leaf mask when SAM2 succeeds; full heatmap fallback when SAM2 fails.
- **Hook Verification**: Verified forward/backward activation hook registration across all 3 crop models.

**Status**: `GRAD_CAM_PASS`
