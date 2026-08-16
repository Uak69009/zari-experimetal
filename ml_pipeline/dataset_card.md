# ZARI.ai Agricultural Image Dataset (153-Class)

## Dataset Description
- **Total Images:** 251211
- **Total Classes:** 153
- **Crop Types:** 21
- **Disease Types:** 102

## Dataset Summary
The ZARI.ai dataset is a highly enriched, massively harmonized agricultural computer vision dataset tailored for disease diagnosis and severity estimation. It merges multiple source datasets (PlantVillage, PlantDoc, NWRD) and provides extensive metadata including domain origins (Lab vs. Field), pathological taxonomies, and OpenCV-computed image quality metrics.

## Supported Tasks
- Multi-class image classification (153 classes)
- Domain adaptation (Lab to Field transfer)
- Disease severity estimation (via segmentation masks)
- Out-of-distribution detection
- Embedding retrieval via FAISS (SigLIP/DINOv2)

## Schema & Features
- `image_path`: Absolute path to the raw image.
- `crop`, `disease`, `class_name`: Taxonomical labels.
- `domain`: Indicates whether the image was taken in a controlled lab setting (`Lab`) or uncontrolled natural environment (`Field`).
- `pathogen_type`: Higher-level categorization (Fungal, Viral, Bacterial, Pest, Nutrient, Healthy).
- `blur_score`, `brightness_score`, `contrast_score`: Computed via OpenCV for quality filtering.

## Known Limitations & Imbalances
- Extreme class imbalance exists (up to 500:1 ratio between majority and minority classes).
- Domain shift is severe (majority of images are `Lab` domain, while actual deployment targets `Field` domain).

## License
Proprietary & Open-Source (Dataset amalgamation governed by original licenses of PlantVillage and PlantDoc. NWRD is proprietary to ZARI.ai).