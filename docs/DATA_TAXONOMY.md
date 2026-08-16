# 📊 ZARI.ai — Data Taxonomy & Ingestion Plan

> This document defines the data ingestion strategy, dataset roles, and the canonical JSON taxonomy structure that unifies all four datasets into a single training-ready format.

---

## 1. Core Datasets

### 1.1 PlantVillage (Phase 1 — Base Pre-training)

| Property | Value |
|---|---|
| **Source** | [GitHub: spMohanty/PlantVillage-Dataset](https://github.com/spMohanty/PlantVillage-Dataset) |
| **Total Images** | ~54,305 |
| **Crops** | 14 crop species, 38 classes |
| **Background** | Controlled lab (single-color background) |
| **Resolution** | 256×256 |
| **Role** | High-volume baseline for transfer learning |

**Filtering Strategy:** Only include crops relevant to the Pakistani region:
- Tomato (10 classes)
- Potato (3 classes)
- Corn/Maize (4 classes)
- Apple (4 classes)
- Grape (4 classes)
- Pepper (2 classes)

### 1.2 PlantDoc (Phase 1 — Noise Introduction)

| Property | Value |
|---|---|
| **Source** | [GitHub: pratikkayal/PlantDoc-Dataset](https://github.com/pratikkayal/PlantDoc-Dataset) |
| **Total Images** | ~2,598 |
| **Crops** | 13 crop species, 27 classes |
| **Background** | Real-world (natural, noisy backgrounds) |
| **Resolution** | Variable |
| **Role** | Break model dependency on clean lab backgrounds |

**Usage:** Merged with filtered PlantVillage in Phase 1 to introduce background diversity.

### 1.3 PlantCity (Phase 2 — Core Domain Fine-Tuning)

| Property | Value |
|---|---|
| **Source** | [Kaggle: codewithsk/plantcity](https://www.kaggle.com/datasets/codewithsk/plantcity-a-comprehensive-images-multicrop-leaves) |
| **Total Images** | 10,667 |
| **Crops** | 12 crop species |
| **Collection Sites** | Charsadda, Chitral (KPK, Pakistan) |
| **Background** | Real Pakistani field conditions |
| **Resolution** | High-resolution field photos |
| **Role** | Primary domain adaptation dataset |

**Usage:** Freeze base layers, aggressively fine-tune classification heads. Apply `Albumentations` field noise augmentations.

### 1.4 NWRD — NUST Wheat Rust Disease (Phase 2 — Specialized Cash Crop)

| Property | Value |
|---|---|
| **Source** | [GitHub: dll-ncai/NUST-Wheat-Rust-Disease-NWRD](https://github.com/dll-ncai/NUST-Wheat-Rust-Disease-NWRD) |
| **Total Images** | ~17,856 |
| **Crops** | Wheat only |
| **Disease Focus** | Leaf Rust, Stem Rust, Stripe Rust, Healthy |
| **Background** | Real field conditions (Pakistan) |
| **Role** | Wheat rust accuracy booster (critical cash crop) |

**Usage:** Merged with PlantCity in Phase 2 training.

---

## 2. Two-Phase Training Strategy

```
                    PHASE 1                              PHASE 2
          ┌──────────────────────┐            ┌──────────────────────┐
          │    PlantVillage      │            │     PlantCity        │
          │   (filtered, ~30K)   │            │   (10,667 images)    │
          │         +            │            │         +            │
          │     PlantDoc         │   ───►     │      NWRD            │
          │    (~2,598)          │  Transfer  │   (~17,856 images)   │
          │                      │            │                      │
          │  Full model training │            │  Freeze base layers  │
          │  Lab → noisy bridge  │            │  Fine-tune heads     │
          │                      │            │  + Albumentations    │
          └──────────────────────┘            └──────────────────────┘
```

---

## 3. Canonical JSON Taxonomy

All four datasets must be mapped to a **single unified taxonomy** before training. This prevents label conflicts (e.g., "Tomato___Early_blight" vs "Tomato Early Blight" vs "tomato_early_blight").

### 3.1 Taxonomy Schema

```json
{
  "version": "1.0.0",
  "total_classes": 0,
  "crops": {
    "<crop_canonical_name>": {
      "crop_id": "<integer>",
      "diseases": {
        "<disease_canonical_name>": {
          "class_id": "<integer (global unique)>",
          "display_name_en": "<English display name>",
          "display_name_ur": "<Urdu display name>",
          "severity_levels": ["mild", "moderate", "severe"],
          "source_labels": {
            "plantvillage": "<original label or null>",
            "plantdoc": "<original label or null>",
            "plantcity": "<original label or null>",
            "nwrd": "<original label or null>"
          }
        }
      }
    }
  }
}
```

### 3.2 Example Taxonomy Entry

```json
{
  "tomato": {
    "crop_id": 1,
    "diseases": {
      "early_blight": {
        "class_id": 1,
        "display_name_en": "Tomato Early Blight",
        "display_name_ur": "ٹماٹر کا ابتدائی جھلسا",
        "severity_levels": ["mild", "moderate", "severe"],
        "source_labels": {
          "plantvillage": "Tomato___Early_blight",
          "plantdoc": "Tomato Early Blight",
          "plantcity": "tomato_early_blight",
          "nwrd": null
        }
      },
      "late_blight": {
        "class_id": 2,
        "display_name_en": "Tomato Late Blight",
        "display_name_ur": "ٹماٹر کا آخری جھلسا",
        "severity_levels": ["mild", "moderate", "severe"],
        "source_labels": {
          "plantvillage": "Tomato___Late_blight",
          "plantdoc": "Tomato Late Blight",
          "plantcity": "tomato_late_blight",
          "nwrd": null
        }
      },
      "healthy": {
        "class_id": 3,
        "display_name_en": "Tomato Healthy",
        "display_name_ur": "ٹماٹر صحت مند",
        "severity_levels": [],
        "source_labels": {
          "plantvillage": "Tomato___healthy",
          "plantdoc": "Tomato Healthy",
          "plantcity": "tomato_healthy",
          "nwrd": null
        }
      }
    }
  },
  "wheat": {
    "crop_id": 2,
    "diseases": {
      "leaf_rust": {
        "class_id": 20,
        "display_name_en": "Wheat Leaf Rust",
        "display_name_ur": "گندم کی پتی کا زنگ",
        "severity_levels": ["mild", "moderate", "severe"],
        "source_labels": {
          "plantvillage": null,
          "plantdoc": null,
          "plantcity": "wheat_leaf_rust",
          "nwrd": "Leaf_Rust"
        }
      },
      "stem_rust": {
        "class_id": 21,
        "display_name_en": "Wheat Stem Rust",
        "display_name_ur": "گندم کے تنے کا زنگ",
        "severity_levels": ["mild", "moderate", "severe"],
        "source_labels": {
          "plantvillage": null,
          "plantdoc": null,
          "plantcity": null,
          "nwrd": "Stem_Rust"
        }
      },
      "stripe_rust": {
        "class_id": 22,
        "display_name_en": "Wheat Stripe Rust (Yellow Rust)",
        "display_name_ur": "گندم کا پیلا زنگ",
        "severity_levels": ["mild", "moderate", "severe"],
        "source_labels": {
          "plantvillage": null,
          "plantdoc": null,
          "plantcity": null,
          "nwrd": "Stripe_Rust"
        }
      },
      "healthy": {
        "class_id": 23,
        "display_name_en": "Wheat Healthy",
        "display_name_ur": "گندم صحت مند",
        "severity_levels": [],
        "source_labels": {
          "plantvillage": null,
          "plantdoc": null,
          "plantcity": "wheat_healthy",
          "nwrd": "Healthy"
        }
      }
    }
  }
}
```

---

## 4. Train / Validation / Test Split Strategy

> **Critical:** The test set must NOT contain any lab-grade data. It must exclusively consist of real Pakistani field images.

| Split | Source | Purpose |
|---|---|---|
| **Train (80%)** | PlantVillage + PlantDoc + PlantCity (80%) + NWRD (80%) | Model learning |
| **Validation (10%)** | PlantCity (10%) + NWRD (10%) | Hyperparameter tuning |
| **Test (10%)** | PlantCity (10%) + NWRD (10%) — **isolated, never seen** | Real-world benchmark |

### Split Rules

1. **No random shuffle across datasets** — PlantVillage/PlantDoc are *never* in the test set.
2. **Stratified split within PlantCity and NWRD** — Ensure every class is represented proportionally.
3. **Patient-level isolation** — If images are from the same plant/field, they must all be in the same split.

---

## 5. Albumentations Augmentation Pipeline (Phase 2)

```python
import albumentations as A

field_augmentation = A.Compose([
    # Geometric
    A.RandomResizedCrop(height=384, width=384, scale=(0.7, 1.0)),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.2),
    A.Rotate(limit=30, p=0.5),

    # Simulated field conditions
    A.MotionBlur(blur_limit=7, p=0.3),
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
    A.RandomSunFlare(src_radius=100, p=0.15),
    A.RandomShadow(p=0.2),
    A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, p=0.1),

    # Color/Contrast shifts
    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
    A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=30, val_shift_limit=20, p=0.4),
    A.CLAHE(clip_limit=4.0, p=0.3),

    # Normalization (ImageNet)
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
```

---

## 6. Data Directory Structure

```
ml_pipeline/
└── data/
    ├── raw/
    │   ├── plantvillage/     ← git clone
    │   ├── plantdoc/         ← git clone
    │   ├── plantcity/        ← kaggle download
    │   └── nwrd/             ← git clone
    └── cleaned/
        ├── taxonomy.json     ← canonical label mapping
        ├── train/
        │   ├── <class_id>_<disease_name>/
        │   └── ...
        ├── val/
        │   ├── <class_id>_<disease_name>/
        │   └── ...
        └── test/
            ├── <class_id>_<disease_name>/
            └── ...
```

---

*Author: Umair Amjad Khan*
