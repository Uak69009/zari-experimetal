# ZARI.ai — Model B Disease Classification Architecture & Audit Specification

**Specification Date**: August 17, 2026  
**Authoritative Manifest**: `ml_pipeline/data/dataset_3crop_final_v4_split.csv` (**49,805 records**)  
**Frozen Model A Checkpoint**: `ml_pipeline/checkpoints/model_a/best_model_a_efficientnetv2_b2.pth` (**Test Acc: 99.50%, Test Macro F1: 0.9926**)  
**Pre-Training Status**: `READY FOR MODEL B SMOKE TEST & TRAINING`  

---

## 1. Executive Summary & Architectural Decisions

| Architectural Dimension | Selected Decision | Rationale |
| :--- | :--- | :--- |
| **Backbone Architecture** | **Pretrained EfficientNetV2-B2** | 7.7M params, 3.52ms latency, native PyTorch Conv2D Grad-CAM, low 1.1GB VRAM, zero overfitting |
| **System Structure** | **Option B: Three Independent Crop-Specific Classifiers** | Zero cross-crop gradient interference, 100% modular, crop-focused feature learning |
| **Tomato Model** | 13 Supervised Classes (35,015 imgs) | Full 13-class supervised classification with Focal Loss ($\gamma=2.0$) |
| **Potato Model** | 3 Supported Supervised Classes (6,458 imgs) + SCRC Fallback | 4 rare classes (<50 imgs) routed to EDL epistemic uncertainty gate |
| **Pepper Model** | 6 Supervised Classes (8,294 imgs) | Full 6-class supervised classification with Focal Loss ($\gamma=2.0$) |
| **Uncertainty Layer** | **Evidential Deep Learning (EDL)** | Dirichlet non-negative Softplus parametrization $\boldsymbol{\alpha} = \text{Softplus}(\mathbf{z}) + 1$ |
| **Selective Risk Layer** | **Selective Classification Risk Control (SCRC)** | Reject low-confidence or high-epistemic-uncertainty queries ($u = K / S \ge u_{\text{thresh}}$) |

---

## 2. Crop-Specific Data Partitions

### A. Tomato Dataset (13 Classes, 35,015 Total Images)

| class_name                    |   train |   val |   test |   total |   train_pct |
|:------------------------------|--------:|------:|-------:|--------:|------------:|
| Tomato_Bacterial_Spot         |    3153 |   394 |    395 |    3942 |        80   |
| Tomato_Early_Blight           |    2475 |   309 |    310 |    3094 |        80   |
| Tomato_Fusarium_Wilt          |     325 |    40 |     42 |     407 |        79.9 |
| Tomato_Healthy                |    2840 |   355 |    355 |    3550 |        80   |
| Tomato_Late_Blight            |    2745 |   343 |    344 |    3432 |        80   |
| Tomato_Leaf_Mold              |    2473 |   309 |    310 |    3092 |        80   |
| Tomato_Miner                  |    1517 |   189 |    191 |    1897 |        80   |
| Tomato_Mosaic_Virus           |     341 |    42 |     44 |     427 |        79.9 |
| Tomato_Septoria_Leaf_Spot     |    2868 |   358 |    360 |    3586 |        80   |
| Tomato_Spider_Mites           |    1850 |   231 |    232 |    2313 |        80   |
| Tomato_Target_Spot            |    1123 |   140 |    141 |    1404 |        80   |
| Tomato_Verticillium_Wilt      |     415 |    51 |     53 |     519 |        80   |
| Tomato_Yellow_Leaf_Curl_Virus |    5881 |   735 |    736 |    7352 |        80   |

### B. Potato Dataset (7 Classes, 6,496 Total Images)

| class_name                |   train |   val |   test |   total |   train_pct |
|:--------------------------|--------:|------:|-------:|--------:|------------:|
| Potato_Bacterial_Soft_Rot |       5 |     1 |      1 |       7 |        71.4 |
| Potato_Early_Blight       |    2194 |   275 |    274 |    2743 |        80   |
| Potato_Healthy            |     936 |   116 |    119 |    1171 |        79.9 |
| Potato_Late_Blight        |    2027 |   253 |    254 |    2534 |        80   |
| Potato_Viral_Leaf_Roll    |      26 |     3 |      4 |      33 |        78.8 |
| Potato_Viral_PVX          |       4 |     1 |      1 |       6 |        66.7 |
| Potato_Viral_PVY          |       1 |     1 |      0 |       2 |        50   |

### C. Pepper Dataset (6 Classes, 8,294 Total Images)

| class_name                  |   train |   val |   test |   total |   train_pct |
|:----------------------------|--------:|------:|-------:|--------:|------------:|
| Pepper_Bacterial_Spot       |    3325 |   420 |    406 |    4151 |        80.1 |
| Pepper_Cercospora_Leaf_Spot |    1238 |   156 |    162 |    1556 |        79.6 |
| Pepper_Healthy              |    1220 |   152 |    153 |    1525 |        80   |
| Pepper_Leaf_Curl            |     340 |    41 |     42 |     423 |        80.4 |
| Pepper_Nutrition_Deficiency |     356 |    43 |     45 |     444 |        80.2 |
| Pepper_Powdery_Mildew       |     156 |    20 |     19 |     195 |        80   |

---

## 3. Computed Class Weights (Train Split Only)

```json
{
  "Tomato_Weights": {
    "Tomato_Yellow_Leaf_Curl_Virus": 0.3663,
    "Tomato_Bacterial_Spot": 0.6833,
    "Tomato_Septoria_Leaf_Spot": 0.7512,
    "Tomato_Healthy": 0.7586,
    "Tomato_Late_Blight": 0.7848,
    "Tomato_Early_Blight": 0.8704,
    "Tomato_Leaf_Mold": 0.8711,
    "Tomato_Spider_Mites": 1.1645,
    "Tomato_Miner": 1.4201,
    "Tomato_Target_Spot": 1.9184,
    "Tomato_Verticillium_Wilt": 5.1911,
    "Tomato_Mosaic_Virus": 6.3176,
    "Tomato_Fusarium_Wilt": 6.6286
  },
  "Potato_Supported_Weights": {
    "Potato_Early_Blight": 0.7835,
    "Potato_Late_Blight": 0.8481,
    "Potato_Healthy": 1.8365
  },
  "Pepper_Weights": {
    "Pepper_Bacterial_Spot": 0.3326,
    "Pepper_Cercospora_Leaf_Spot": 0.8932,
    "Pepper_Healthy": 0.9064,
    "Pepper_Nutrition_Deficiency": 3.1063,
    "Pepper_Leaf_Curl": 3.2525,
    "Pepper_Powdery_Mildew": 7.0887
  }
}
```

---

## 4. Final Model B Architecture Specification

```text
============================================================
MODEL_B_FINAL_SPECIFICATION
============================================================
Model A                 : FROZEN EfficientNetV2-B2 (Test Acc: 99.50%, F1: 0.9926)
Model B Backbone        : Pretrained EfficientNetV2-B2
Model B Structure       : 3 Independent Crop-Specific Classifiers
Tomato Outputs          : 13 Supervised Classes
Potato Outputs          : 3 Supported Supervised Classes (Early Blight, Late Blight, Healthy)
Pepper Outputs          : 6 Supervised Classes
Rare Potato Handling    : Tier D Insufficient-Evidence EDL Epistemic Uncertainty Fallback
Loss                    : Focal Loss (gamma=2.0) with Inverse-Frequency Class Weights
Class Weight Formula    : w_c = clip(N_tr / (K * n_c), 0.1, 10.0) [Train Split Only]
Sampler                 : Standard PyTorch RandomSampler (Zero artificial duplication)
Input                   : 256x256 RGB Tensor [B, 3, 256, 256]
Optimizer               : AdamW (weight_decay=1e-4)
Backbone LR             : 1e-4 (Differential Fine-Tuning)
Head LR                 : 1e-3 (Head Warmup)
Weight Decay            : 1e-4
Dropout                 : p=0.30
Scheduler               : ReduceLROnPlateau(mode='min', factor=0.5, patience=2, min_lr=1e-7)
Early Stopping          : Validation Macro F1 (patience=5, min_delta=0.001)
EDL                     : Built into Classifier Head (Softplus Dirichlet parametrization)
SCRC                    : Selective Risk Control Rejection (u = K / S >= u_thresh)
Grad-CAM                : Native PyTorch Conv2D Layer (features.7.1)
MODEL_B_PRETRAINING_STATUS: READY
============================================================
```