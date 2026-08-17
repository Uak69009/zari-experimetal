# ZARI.ai — Class Imbalance & Class Weights Analysis Report (V3 Dataset)

**Analysis Date**: August 16, 2026  
**Dataset Manifest**: `ml_pipeline/data/dataset_final_training_v3_clean.csv`  
**Total Images Analyzed**: **49,805**  
**Crops Analyzed**: **Tomato, Potato, Pepper** (26 total classes)  

---

## 1. Dataset Overview & Crop Volume

| Crop | Total Images | Canonical Classes | Train Split | Val Split | Test Split |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Tomato** | **35,015** | 13 | 28,001 | 3,477 | 3,537 |
| **Potato** | **6,496** | 7 | 5,202 | 689 | 605 |
| **Pepper** | **8,294** | 6 | 6,638 | 812 | 844 |

---

## 2. Crop-Level Imbalance Statistics

| Crop | Min Class Count | Max Class Count | Mean Class Count | Imbalance Ratio (Max / Min) |
| :--- | :---: | :---: | :---: | :---: |
| **Tomato** | 407 | 7,352 | 2693.5 | **18.1x** |
| **Potato** | 2 | 2,743 | 928.0 | **1371.5x** |
| **Pepper** | 195 | 4,151 | 1382.3 | **21.3x** |

---

## 3. Full Per-Class Distribution & Computed Weights

The class weights are computed per crop using the standard inverse-frequency formula:
$$w_i = \text{clip}\left(\frac{N}{K \cdot n_i}, 0.1, 10.0\right)$$

### Tomato Classes (13 Classes)

| Class Name | Image Count | Crop Percentage | Computed Class Weight |
| :--- | :---: | :---: | :---: |
| `Tomato_Fusarium_Wilt` | **407** | 1.16% | **3.1054** |
| `Tomato_Mosaic_Virus` | **427** | 1.22% | **2.9599** |
| `Tomato_Verticillium_Wilt` | **519** | 1.48% | **2.4352** |
| `Tomato_Target_Spot` | **1,404** | 4.01% | **0.9002** |
| `Tomato_Miner` | **1,897** | 5.42% | **0.6663** |
| `Tomato_Spider_Mites` | **2,313** | 6.61% | **0.5464** |
| `Tomato_Leaf_Mold` | **3,092** | 8.83% | **0.4088** |
| `Tomato_Early_Blight` | **3,094** | 8.84% | **0.4085** |
| `Tomato_Late_Blight` | **3,432** | 9.80% | **0.3683** |
| `Tomato_Healthy` | **3,550** | 10.14% | **0.3560** |
| `Tomato_Septoria_Leaf_Spot` | **3,586** | 10.24% | **0.3525** |
| `Tomato_Bacterial_Spot` | **3,942** | 11.26% | **0.3206** |
| `Tomato_Yellow_Leaf_Curl_Virus` | **7,352** | 21.00% | **0.1719** |

### Potato Classes (7 Classes)

| Class Name | Image Count | Crop Percentage | Computed Class Weight |
| :--- | :---: | :---: | :---: |
| `Potato_Viral_PVY` | **2** | 0.03% | **4.1595** |
| `Potato_Viral_PVX` | **6** | 0.09% | **1.3865** |
| `Potato_Bacterial_Soft_Rot` | **7** | 0.11% | **1.1884** |
| `Potato_Viral_Leaf_Roll` | **33** | 0.51% | **0.2521** |
| `Potato_Healthy` | **1,171** | 18.03% | **0.1000** |
| `Potato_Late_Blight` | **2,534** | 39.01% | **0.1000** |
| `Potato_Early_Blight` | **2,743** | 42.23% | **0.1000** |

### Pepper Classes (6 Classes)

| Class Name | Image Count | Crop Percentage | Computed Class Weight |
| :--- | :---: | :---: | :---: |
| `Pepper_Powdery_Mildew` | **195** | 2.35% | **2.7268** |
| `Pepper_Leaf_Curl` | **423** | 5.10% | **1.2571** |
| `Pepper_Nutrition_Deficiency` | **444** | 5.35% | **1.1976** |
| `Pepper_Healthy` | **1,525** | 18.39% | **0.3487** |
| `Pepper_Cercospora_Leaf_Spot` | **1,556** | 18.76% | **0.3417** |
| `Pepper_Bacterial_Spot` | **4,151** | 50.05% | **0.1281** |

