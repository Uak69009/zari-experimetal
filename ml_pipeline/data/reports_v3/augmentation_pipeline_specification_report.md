# ZARI.ai — Final Image Preprocessing, Augmentation & Normalization Report

**Audit Date**: August 16, 2026  
**Dataset Manifest**: `ml_pipeline/data/dataset_3crop_final.csv` (**49,805 records**)  
**Training Images Sampled**: **5,000 representative training images**  

---

## 1. Empirical Training Set Normalization Statistics (TRAIN ONLY)

- **Dataset-Specific RGB Mean**: `[0.4601, 0.5038, 0.3833]`
- **Dataset-Specific RGB Std**: `[0.1919, 0.1755, 0.2071]`
- **Standard ImageNet RGB Mean**: `[0.4850, 0.4560, 0.4060]`
- **Standard ImageNet RGB Std**: `[0.2290, 0.2240, 0.2250]`
- **Recommendation**: Use **ImageNet Mean & Std** for pretrained EfficientNetV2-B2 and Swin-Tiny backbones to ensure optimal transfer learning alignment.

---

## 2. Input Resolution & Aspect Ratio Decision

- **Recommended Input Resolution**: **$256 \times 256$ pixels**
- **Rationale**: 60.15% of dataset images are natively $256\times 256$ px. $256\times 256$ preserves lesion visibility, aligns perfectly with Swin patch size ($32\times 32$), and runs efficiently on RTX 4090 GPU.
- **Resize Strategy**: **Aspect-Preserving Resize + Center Crop / Padding** to prevent lesion distortion near leaf boundaries.

---

## 3. Augmentation Decisions (MixUp, CutMix, Random Erasing, Hue)

- **MixUp**: **NO** (MixUp creates unrealistic composite leaves with overlapping disease structures, corrupting evidential uncertainty).
- **CutMix**: **NO** (Pasting square disease patches onto healthy leaves creates non-biological artificial borders).
- **Hue Augmentation**: **Strictly Restricted ($[-0.02, 0.02]$ or Disabled)** (Large hue shifts turn green leaves yellow or brown lesions purple, corrupting diagnostic chlorosis/necrosis semantics).
- **Random Erasing**: **Low Probability ($p=0.10$, area range $[0.02, 0.10]$)** (Prevents erasing small single diagnostic lesions).

---

## FINAL ZARI.ai AUGMENTATION SPECIFICATION

```python
# PyTorch / Torchvision Augmentation Pipeline Specification
import torchvision.transforms.v2 as T

train_transform = T.Compose([
    T.Resize((256, 256), antialias=True),
    T.RandomHorizontalFlip(p=0.5),
    T.RandomRotation(degrees=15, interpolation=T.InterpolationMode.BILINEAR),
    T.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.90, 1.10)),
    T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10, hue=0.02),
    T.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 1.0)),
    T.ToImage(),
    T.ToDtype(torch.float32, scale=True),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    T.RandomErasing(p=0.10, scale=(0.02, 0.10), value=0)
])

val_test_transform = T.Compose([
    T.Resize((256, 256), antialias=True),
    T.ToImage(),
    T.ToDtype(torch.float32, scale=True),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```