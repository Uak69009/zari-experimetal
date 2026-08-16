# ZARI.ai Model Architecture & Training Implementation Plan

Based on the highly advanced research specification document you provided, this plan outlines the architecture for the **ZARI.ai Neural Network** and its training engine. This design targets publication-level novelty by integrating domain-stratified learning, coordinate attention, and uncertainty estimation.

## User Review Required

> [!IMPORTANT]
> **Evidential Deep Learning (Dirichlet Head):** Instead of standard Softmax, the network will output Dirichlet concentration parameters ($\alpha$). This is highly advanced and will require a custom Loss Function (Evidential NLL + KL Divergence regularizer) in the training loop.
> 
> **Coordinate Attention:** We will inject a custom Coordinate Attention module directly after the EfficientNet backbone to sharpen the model's focus on lesions rather than backgrounds.

## Open Questions

1. **Backbone Size:** I propose using `tf_efficientnetv2_s` (Small) via the `timm` library, as it strikes a perfect balance between mobile NPU deployability and high accuracy. Does this work for you, or would you prefer the ultra-light `B0`?
2. **Domain-Stratified Batching:** Do you want me to write a custom PyTorch `Sampler` in the dataloader that forces exactly a 50/50 mix of Lab and Field images in every batch?

## Proposed Changes

### Neural Network Architecture
#### [NEW] [ml_pipeline/04_model.py](file:///d:/New%20folder/zari/zari-ai/ml_pipeline/04_model.py)
This file will contain the PyTorch model definition.
- **`CoordAtt` Class:** A custom Coordinate Attention block embedding 1D spatial encodings into channel attention.
- **`ZariNet` Class:**
  - **Backbone:** Loads a pretrained `efficientnetv2_s` from `timm` (stripping its original classification head).
  - **Attention Injection:** Passes the pooled features through our `CoordAtt` block.
  - **Shared Embedding:** An MLP layer mapping features down to a 256-D embedding space.
  - **Evidential Head:** A final linear layer outputting 153 values. A `Softplus` or `ReLU` activation is applied to ensure all $\alpha_i > 0$ for the Dirichlet distribution.

### Training Engine & Loss Functions
#### [NEW] [ml_pipeline/05_train.py](file:///d:/New%20folder/zari/zari-ai/ml_pipeline/05_train.py)
This script will train the `ZariNet`.
- **Custom Evidential Loss (`edl_loss`):** Computes the expected Mean Squared Error (or NLL) of the Dirichlet distribution against the one-hot targets, plus a KL-divergence regularizer to penalize false certainty on unknown distributions.
- **Optimizer & Scheduler:** Uses `AdamW` with decoupled weight decay, paired with a `CosineAnnealingLR` scheduler with warmup.
- **Metrics Tracking:** Tracks Top-1 Accuracy, Macro-F1 (vital for the 500:1 class imbalances), and measures average Evidential Uncertainty (Sum of $\alpha$).

## Verification Plan

### Automated Tests
- I will run a dummy tensor (e.g., shape `[8, 3, 224, 224]`) through `04_model.py` to mathematically verify that the output shape is exactly `[8, 153]` and that all output values are strictly positive ($\ge 1.0$) for Dirichlet concentration.

### Manual Verification
- You will be able to review the model's forward pass logic and the custom Evidential Loss function implementation to ensure it matches the R-EDL literature before we commence actual training.
