# Master Dataset V3 — Dataset Cleaning Report

**Report Date**: August 16, 2026  
**Cleaned CSV Path**: `ml_pipeline/data/dataset_final_training_v3_clean.csv`  

---

## 1. Summary of Cleaning Actions

| Cleaning Check | Issues Found | Action Taken |
| :--- | :---: | :--- |
| **Corrupt Files** | **0** | Removed unopenable images |
| **Tiny Images (<50px)** | **0** | Removed low-resolution images |
| **Exact SHA256 Duplicates** | **161** | Dropped duplicate samples |
| **Unmapped Unknown Classes** | **124** | Filtered uninformative classes |
| **Taxonomy Alias Inconsistencies** | **13800** | Consolidated into canonical class names |

---

## 2. Before vs After Dataset Volumes

- **Initial Dataset V3 Volume**: **49,517** images across 32 classes
- **Total Images Removed / Fixed**: **285** images
- **Final Cleaned Dataset Volume**: **49,232** images across **26** classes

### Cleaned Crop Breakdown

| Crop Name | Before Cleaning | After Cleaning | Net Delta |
| :--- | :---: | :---: | :---: |
| **Pepper** | 7,799 | **7,737** | -62 |
| **Potato** | 6,501 | **6,496** | -5 |
| **Tomato** | 35,217 | **34,999** | -218 |

---

## 3. Final Cleaned Class Registry (26 Classes)


### Pepper Classes (6 classes)

| Class Name | Image Count | Percentage |
| :--- | :---: | :---: |
| `Pepper_Bacterial_Spot` | 3,594 | 46.45% |
| `Pepper_Cercospora_Leaf_Spot` | 1,556 | 20.11% |
| `Pepper_Healthy` | 1,525 | 19.71% |
| `Pepper_Nutrition_Deficiency` | 444 | 5.74% |
| `Pepper_Leaf_Curl` | 423 | 5.47% |
| `Pepper_Powdery_Mildew` | 195 | 2.52% |

### Potato Classes (7 classes)

| Class Name | Image Count | Percentage |
| :--- | :---: | :---: |
| `Potato_Early_Blight` | 2,743 | 42.23% |
| `Potato_Late_Blight` | 2,534 | 39.01% |
| `Potato_Healthy` | 1,171 | 18.03% |
| `Potato_Viral_Leaf_Roll` | 33 | 0.51% |
| `Potato_Bacterial_Soft_Rot` | 7 | 0.11% |
| `Potato_Viral_PVX` | 6 | 0.09% |
| `Potato_Viral_PVY` | 2 | 0.03% |

### Tomato Classes (13 classes)

| Class Name | Image Count | Percentage |
| :--- | :---: | :---: |
| `Tomato_Yellow_Leaf_Curl_Virus` | 7,352 | 21.01% |
| `Tomato_Bacterial_Spot` | 3,942 | 11.26% |
| `Tomato_Septoria_Leaf_Spot` | 3,586 | 10.25% |
| `Tomato_Healthy` | 3,534 | 10.10% |
| `Tomato_Late_Blight` | 3,432 | 9.81% |
| `Tomato_Early_Blight` | 3,094 | 8.84% |
| `Tomato_Leaf_Mold` | 3,092 | 8.83% |
| `Tomato_Spider_Mites` | 2,313 | 6.61% |
| `Tomato_Miner` | 1,897 | 5.42% |
| `Tomato_Target_Spot` | 1,404 | 4.01% |
| `Tomato_Verticillium_Wilt` | 519 | 1.48% |
| `Tomato_Mosaic_Virus` | 427 | 1.22% |
| `Tomato_Fusarium_Wilt` | 407 | 1.16% |
