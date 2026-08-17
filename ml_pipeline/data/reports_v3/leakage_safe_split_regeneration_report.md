# ZARI.ai — Leakage-Safe Split Regeneration Report

**Audit Date**: August 17, 2026  
**Master Manifest**: `ml_pipeline/data/dataset_3crop_final.csv` (**49,805 records**)  
**Versioned Manifest**: `ml_pipeline/data/dataset_3crop_final_v4_split.csv`  
**Status**: `SAFE — ZERO LEAKAGE VERIFIED`  

---

## 1. Executive Summary

- **Master Dataset Rows**: **49,805 images** (100% Unchanged)
- **Total Image Families Constructed**: **49,365 families** (via DSU chaining SHA-256 + pHash $h \le 2$)
- **Exact SHA-256 Leakage**: **`0 Hashes`**
- **pHash $h=0$ Leakage**: **`0 Pairs`**
- **pHash $h \le 2$ Family Leakage**: **`0 Families`**

---

## 2. Image Family Structure Statistics

- **Total Image Families**: **49,088**
- **Singleton Families**: **48,401** (98.6%)
- **Multi-Image Families**: **687** (1.4%)
- **Largest Family Size**: **5 images**
- **Families Multi-Source**: **0**
- **Families Multi-Class**: **1**

---

## 3. Old vs. New Split Comparison

| Metric | Old Split (SHA-256 Only) | New Split (DSU Family-Atomic) | Delta / Change |
| :--- | :---: | :---: | :---: |
| **Train Split Count** | 39,837 (80.0%) | **39,834** (79.98%) | -3 |
| **Validation Split Count** | 4,969 (10.0%) | **4,978** (9.99%) | +9 |
| **Test Split Count** | 4,999 (10.0%) | **4,993** (10.03%) | -6 |
| **pHash h=0 Cross-Split Pairs** | 199 pairs | **0 pairs** | **-199 pairs (Resolved)** |
| **Image Family Cross-Split Leakage** | 193 families | **0 families** | **-193 families (Resolved)** |
| **Changed Image Memberships** | — | **5,108 images** | 10.26% |