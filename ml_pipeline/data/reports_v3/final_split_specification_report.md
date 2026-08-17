# ZARI.ai — Final Leakage-Safe Split Audit & Specification Report

**Audit Date**: August 16, 2026  
**Master Manifest**: `ml_pipeline/data/dataset_3crop_final.csv` (**49,805 records**)  
**Scope**: **STRICTLY 3 CROPS** (Tomato, Potato, Bell Pepper)  

---

## 1. Existing Split Verification & Leakage Results

- **Split Script**: `ml_pipeline/scripts/v3/build_v3_dataset.py` (Fixed seed: `42`)
- **Grouping Mechanism**: `image_group_id` = `grp_{sha256[:12]}`
- **Exact SHA-256 Cross-Split Leakage**: **`0 Hashes`**
  - Train ∩ Validation Overlap : **0**
  - Train ∩ Test Overlap       : **0**
  - Validation ∩ Test Overlap  : **0**

---

## 2. Crop & Source Distribution Across Splits

### Crop Distribution

| crop   |   test |   train |   val |   All |
|:-------|-------:|--------:|------:|------:|
| Pepper |    844 |    6638 |   812 |  8294 |
| Potato |    605 |    5202 |   689 |  6496 |
| Tomato |   3537 |   28001 |  3477 | 35015 |
| All    |   4986 |   39841 |  4978 | 49805 |

### Source Dataset Distribution

| source_dataset                                     |   test |   train |   val |
|:---------------------------------------------------|-------:|--------:|------:|
| Pepper Bell Leaf Disease                           |     56 |     446 |    55 |
| Tomato Leaf Disease Classification Dataset in Paki |      2 |      12 |     2 |
| bell_pepper_mendeley                               |    530 |    4155 |   506 |
| plantcity                                          |   1561 |   12500 |  1581 |
| plantdoc                                           |     93 |     768 |   106 |
| plantvillage                                       |   2322 |   18189 |  2262 |
| potato_bangladesh                                  |      4 |      53 |    15 |
| potato_pld                                         |    365 |    3291 |   400 |
| tomato_pakistan                                    |     53 |     427 |    51 |

---

## 3. Supervised vs Total Dataset Membership Summary

| Split Category | Total Dataset Membership (26 Classes) | Supervised Loss Membership (22 Classes, Excl <50) |
| :--- | :---: | :---: |
| **Train (80%)** | **39,841** | **39,800** |
| **Validation (10%)** | **4,978** | **4,973** |
| **Test (10%)** | **4,986** | **4,984** |
| **TOTAL** | **49,805** | **49,757** |

---

## FINAL SPLIT SPECIFICATION

```text
Split Ratios      : 80% Train / 10% Validation / 10% Test
Random Seed       : 42 (fixed, deterministic, reproducible)
Grouping Key      : image_group_id = grp_{sha256[:12]}
Stratification Key: canonical_class_name
Handling Tier D   : 4 Potato classes (<50 imgs) retained for SCRC evaluation pool, excluded from CE loss
Field/Lab Shift   : Preserved (66.9% Potato Field, 68.0% Pepper Natural across all splits)
SHA-256 Leakage   : 0 Hashes (Verified)
```