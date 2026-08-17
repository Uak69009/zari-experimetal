# ZARI.ai — Production Operations & Reproducibility Runbook

## Execution Commands

### 1. Run Complete Inference Pipeline
```bash
python3 ml_pipeline/scripts/v3/train_full_model_b.py
```

### 2. Validate Checkpoint Checksums
```bash
sha256sum ml_pipeline/checkpoints/model_a/best_model_a_efficientnetv2_b2.pth
sha256sum ml_pipeline/checkpoints/model_b/best_model_b_*.pth
```
