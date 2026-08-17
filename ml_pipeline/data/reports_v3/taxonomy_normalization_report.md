# ZARI.ai — 3-Crop Taxonomy & Class Normalization Audit Report

**Audit Date**: August 16, 2026  
**Master Manifest**: `ml_pipeline/data/dataset_3crop_final.csv` (**49,805 records**)  
**Scope**: **STRICTLY 3 CROPS** (Tomato, Potato, Bell Pepper)  
**Verification Method**: 100% Empirical CSV Analysis (Zero modifications)  

---

## 1. Scope & Crop Verification

- **Total Records Inspected**: **49,805 images**
- **Crops Present**: **Tomato** (35,015 images), **Pepper** (8,294 images), **Potato** (6,496 images)
- **Non-Target Crop Leakage**: **`0 records`** (100% of rows belong strictly to the 3 target crops).

---

## 2. Current Class Inventory (26 Classes)

| Exact Class Label | Target Crop | Disease / Condition | Total Images | Source Count | Source Datasets |
| :--- | :---: | :--- | :---: | :---: | :--- |
| `Pepper_Bacterial_Spot` | Pepper | Bacterial_Spot | **4,151** | 4 | Pepper Bell Leaf Disease, bell_pepper_mendeley, plantdoc, plantvillage |
| `Pepper_Cercospora_Leaf_Spot` | Pepper | Cercospora_Leaf_Spot | **1,556** | 1 | bell_pepper_mendeley |
| `Pepper_Healthy` | Pepper | Healthy | **1,525** | 2 | bell_pepper_mendeley, plantvillage |
| `Pepper_Leaf_Curl` | Pepper | Leaf_Curl | **423** | 1 | bell_pepper_mendeley |
| `Pepper_Nutrition_Deficiency` | Pepper | Nutrition_Deficiency | **444** | 1 | bell_pepper_mendeley |
| `Pepper_Powdery_Mildew` | Pepper | Powdery_Mildew | **195** | 1 | bell_pepper_mendeley |
| `Potato_Bacterial_Soft_Rot` | Potato | Bacterial_Soft_Rot | **7** | 1 | potato_bangladesh |
| `Potato_Early_Blight` | Potato | Early_Blight | **2,743** | 3 | plantdoc, plantvillage, potato_pld |
| `Potato_Healthy` | Potato | Healthy | **1,171** | 3 | plantvillage, potato_bangladesh, potato_pld |
| `Potato_Late_Blight` | Potato | Late_Blight | **2,534** | 4 | plantdoc, plantvillage, potato_bangladesh, potato_pld |
| `Potato_Viral_Leaf_Roll` | Potato | Viral_Leaf_Roll | **33** | 1 | potato_bangladesh |
| `Potato_Viral_PVX` | Potato | Viral_PVX | **6** | 1 | potato_bangladesh |
| `Potato_Viral_PVY` | Potato | Viral_PVY | **2** | 1 | potato_bangladesh |
| `Tomato_Bacterial_Spot` | Tomato | Bacterial_Spot | **3,942** | 3 | plantcity, plantdoc, plantvillage |
| `Tomato_Early_Blight` | Tomato | Early_Blight | **3,094** | 4 | plantcity, plantdoc, plantvillage, tomato_pakistan |
| `Tomato_Fusarium_Wilt` | Tomato | Fusarium_Wilt | **407** | 1 | plantcity |
| `Tomato_Healthy` | Tomato | Healthy | **3,550** | 4 | Tomato Leaf Disease Classification Dataset in Paki, plantcity, plantvillage, tomato_pakistan |
| `Tomato_Late_Blight` | Tomato | Late_Blight | **3,432** | 4 | plantcity, plantdoc, plantvillage, tomato_pakistan |
| `Tomato_Leaf_Mold` | Tomato | Leaf_Mold | **3,092** | 4 | plantcity, plantdoc, plantvillage, tomato_pakistan |
| `Tomato_Miner` | Tomato | Miner | **1,897** | 1 | plantcity |
| `Tomato_Mosaic_Virus` | Tomato | Mosaic_Virus | **427** | 2 | plantdoc, plantvillage |
| `Tomato_Septoria_Leaf_Spot` | Tomato | Septoria_Leaf_Spot | **3,586** | 4 | plantcity, plantdoc, plantvillage, tomato_pakistan |
| `Tomato_Spider_Mites` | Tomato | Spider_Mites | **2,313** | 3 | plantcity, plantdoc, plantvillage |
| `Tomato_Target_Spot` | Tomato | Target_Spot | **1,404** | 1 | plantvillage |
| `Tomato_Verticillium_Wilt` | Tomato | Verticillium_Wilt | **519** | 1 | plantcity |
| `Tomato_Yellow_Leaf_Curl_Virus` | Tomato | Yellow_Leaf_Curl_Virus | **7,352** | 4 | plantcity, plantdoc, plantvillage, tomato_pakistan |

---

## 3. FINAL CANONICAL TAXONOMY

### 🍅 Tomato (13 Classes — 35,015 Images)
```text
Tomato
  ├── TOMATO_BACTERIAL_SPOT             (3,942 images)
  ├── TOMATO_EARLY_BLIGHT               (3,094 images)
  ├── TOMATO_FUSARIUM_WILT                (407 images)
  ├── TOMATO_HEALTHY                    (3,550 images)
  ├── TOMATO_LATE_BLIGHT                (3,432 images)
  ├── TOMATO_LEAF_MOLD                  (3,092 images)
  ├── TOMATO_MINER                      (1,897 images)
  ├── TOMATO_MOSAIC_VIRUS                 (427 images)
  ├── TOMATO_SEPTORIA_LEAF_SPOT         (3,586 images)
  ├── TOMATO_SPIDER_MITES               (2,313 images)
  ├── TOMATO_TARGET_SPOT                (1,404 images)
  ├── TOMATO_VERTICILLIUM_WILT            (519 images)
  └── TOMATO_YELLOW_LEAF_CURL_VIRUS     (7,352 images)
```

### 🥔 Potato (7 Classes — 6,496 Images)
```text
Potato
  ├── POTATO_BACTERIAL_SOFT_ROT             (7 images)
  ├── POTATO_EARLY_BLIGHT               (2,743 images)
  ├── POTATO_HEALTHY                    (1,171 images)
  ├── POTATO_LATE_BLIGHT                (2,534 images)
  ├── POTATO_VIRAL_LEAF_ROLL               (33 images)
  ├── POTATO_VIRAL_PVX                      (6 images)
  └── POTATO_VIRAL_PVY                      (2 images)
```

### 🫑 Bell Pepper (6 Classes — 8,294 Images)
```text
Pepper
  ├── PEPPER_BACTERIAL_SPOT             (4,151 images)
  ├── PEPPER_CERCOSPORA_LEAF_SPOT       (1,556 images)
  ├── PEPPER_HEALTHY                    (1,525 images)
  ├── PEPPER_LEAF_CURL                    (423 images)
  ├── PEPPER_NUTRITION_DEFICIENCY         (444 images)
  └── PEPPER_POWDERY_MILDEW               (195 images)
```

---

## 4. CLASS-NORMALIZATION DECISION TABLE

| Class Name | Target Crop | Proposed Canonical ID | Status | Notes |
| :--- | :---: | :--- | :---: | :--- |
| `Tomato_Bacterial_Spot` | Tomato | `TOMATO_BACTERIAL_SPOT` | **MUST REMAIN SEPARATE** | Distinct bacterial disease (*Xanthomonas*) |
| `Tomato_Early_Blight` | Tomato | `TOMATO_EARLY_BLIGHT` | **MUST REMAIN SEPARATE** | Distinct fungal disease (*Alternaria solani*) |
| `Tomato_Fusarium_Wilt` | Tomato | `TOMATO_FUSARIUM_WILT` | **MUST REMAIN SEPARATE** | Distinct vascular fungus (*Fusarium oxysporum*) |
| `Tomato_Healthy` | Tomato | `TOMATO_HEALTHY` | **MUST REMAIN SEPARATE** | Non-diseased leaf control |
| `Tomato_Late_Blight` | Tomato | `TOMATO_LATE_BLIGHT` | **MUST REMAIN SEPARATE** | Distinct oomycete (*Phytophthora infestans*) |
| `Tomato_Leaf_Mold` | Tomato | `TOMATO_LEAF_MOLD` | **MUST REMAIN SEPARATE** | Distinct fungal disease (*Passalora fulva*) |
| `Tomato_Miner` | Tomato | `TOMATO_MINER` | **MUST REMAIN SEPARATE** | Distinct pest damage (*Tuta absoluta*) |
| `Tomato_Mosaic_Virus` | Tomato | `TOMATO_MOSAIC_VIRUS` | **MUST REMAIN SEPARATE** | Distinct viral pathogen (ToMV) |
| `Tomato_Septoria_Leaf_Spot` | Tomato | `TOMATO_SEPTORIA_LEAF_SPOT` | **MUST REMAIN SEPARATE** | Distinct fungal disease (*Septoria lycopersici*) |
| `Tomato_Spider_Mites` | Tomato | `TOMATO_SPIDER_MITES` | **MUST REMAIN SEPARATE** | Distinct arachnid pest (*Tetranychus urticae*) |
| `Tomato_Target_Spot` | Tomato | `TOMATO_TARGET_SPOT` | **MUST REMAIN SEPARATE** | Distinct fungal disease (*Corynespora cassiicola*) |
| `Tomato_Verticillium_Wilt` | Tomato | `TOMATO_VERTICILLIUM_WILT` | **MUST REMAIN SEPARATE** | Distinct vascular fungus (*Verticillium dahliae*) |
| `Tomato_Yellow_Leaf_Curl_Virus` | Tomato | `TOMATO_YELLOW_LEAF_CURL_VIRUS` | **SAFE TO MERGE** | Consolidated with `Tomato_Curl` (TYLCV virus) |
| `Potato_Bacterial_Soft_Rot` | Potato | `POTATO_BACTERIAL_SOFT_ROT` | **NEEDS MANUAL REVIEW / EXCLUSION** | Thin class (7 images) |
| `Potato_Early_Blight` | Potato | `POTATO_EARLY_BLIGHT` | **MUST REMAIN SEPARATE** | Major fungal disease (*Alternaria solani*) |
| `Potato_Healthy` | Potato | `POTATO_HEALTHY` | **MUST REMAIN SEPARATE** | Non-diseased control |
| `Potato_Late_Blight` | Potato | `POTATO_LATE_BLIGHT` | **MUST REMAIN SEPARATE** | Major oomycete (*Phytophthora infestans*) |
| `Potato_Viral_Leaf_Roll` | Potato | `POTATO_VIRAL_LEAF_ROLL` | **NEEDS MANUAL REVIEW / EXCLUSION** | Thin class (33 images) |
| `Potato_Viral_PVX` | Potato | `POTATO_VIRAL_PVX` | **NEEDS MANUAL REVIEW / EXCLUSION** | Thin class (6 images) |
| `Potato_Viral_PVY` | Potato | `POTATO_VIRAL_PVY` | **NEEDS MANUAL REVIEW / EXCLUSION** | Thin class (2 images) |
| `Pepper_Bacterial_Spot` | Pepper | `PEPPER_BACTERIAL_SPOT` | **MUST REMAIN SEPARATE** | Major bacterial disease (*Xanthomonas*) |
| `Pepper_Cercospora_Leaf_Spot` | Pepper | `PEPPER_CERCOSPORA_LEAF_SPOT` | **MUST REMAIN SEPARATE** | Distinct fungal disease (*Cercospora capsici*) |
| `Pepper_Healthy` | Pepper | `PEPPER_HEALTHY` | **MUST REMAIN SEPARATE** | Non-diseased control |
| `Pepper_Leaf_Curl` | Pepper | `PEPPER_LEAF_CURL` | **MUST REMAIN SEPARATE** | Distinct viral disease (ChiLCV) |
| `Pepper_Nutrition_Deficiency` | Pepper | `PEPPER_NUTRITION_DEFICIENCY` | **MUST REMAIN SEPARATE** | Abiotic disorder |
| `Pepper_Powdery_Mildew` | Pepper | `PEPPER_POWDERY_MILDEW` | **MUST REMAIN SEPARATE** | Distinct fungal disease (*Leveillula taurica*) |

---

## 5. CRITICAL SAFETY VERIFICATION

- **Current Master CSV Total Images**: **49,805**
- **Sum of Canonical Class Images**: **49,805**
- **Verification Status**: ✅ **100% MATCH CONFIRMED** (`SUM = 49,805`)
