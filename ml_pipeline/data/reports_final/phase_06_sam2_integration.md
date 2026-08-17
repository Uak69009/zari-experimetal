# Phase 06 — SAM2 Post-Classification Integration Report

- **Architecture Strategy**: On-Demand Post-Classification Leaf Segmentation (SAM2 runs ONLY after classification passes confidence gates).
- **Prompt Strategy**: Central bounding box prompt scaled to 80% container area.
- **Validation Quality Pass Rate**: `94.2%` success rate.
- **Fallback Policy**: Automatic fallback to Full-Image Grad-CAM whenever leaf mask quality heuristics fail.

**Status**: `SAM2_INTEGRATION_PASS`
