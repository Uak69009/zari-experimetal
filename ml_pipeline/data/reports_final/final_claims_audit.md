# ZARI.ai — Production Claims Verification Audit Report

**Audit Date**: August 17, 2026  
**Audit Result**: **`PRODUCTION_READY_WITH_LIMITATIONS`**  

---

## 1. Audit Summary & Claim Classification

| Claim Area | Stated Claim in Previous Report | Audit Classification | Evidence & Corrected Scientific Wording |
| :--- | :--- | :---: | :--- |
| **Model Accuracy** | Model A 99.50%, Model B Tomato 98.26%, Potato 96.75%, Pepper 99.40% | **SUPPORTED** | Confirmed on locked test split (4,993 images). |
| **SCRC Selective Risk** | Target Selective Risk $<2.0\%$ | **SUPPORTED** | Calibrated on validation split. Standard gate achieves **`1.04%` Selective Risk** @ **`98.96%` Selective Acc**. |
| **SAM2 Performance** | "SAM2 Quality Pass = 94.2%" | **PARTIALLY SUPPORTED** | 94.2% applies to **Laboratory images only**. Overall mixed dataset pass rate is **`83.3%`** (Field pass = `72.6%`, Fallback rate = `16.7%`). |
| **Severity Estimation** | "Visual Disease Coverage" | **PARTIALLY SUPPORTED** | Heuristic visual proxy. Stated strictly as `"Estimated Visual Disease Coverage Proxy"`; not biological infection rate. |
| **RAG System** | "0 Hallucination Risk" | **PARTIALLY SUPPORTED** | Replaced with: `"22 verified agronomic entries; no unsupported claims were detected in the evaluated RAG test suite."` |
| **Pipeline Latency** | "5.12 ms End-to-End Latency" | **PARTIALLY SUPPORTED** | 5.12 ms is **GPU neural forward pass only**. Total end-to-end execution including decode, SAM2, Grad-CAM, and RAG is **`~43.0 ms`**. |

---

## SAM2 Metric Reconciliation
- **Sample 1 (83.3% Overall)**: Evaluated across 1,000 mixed dataset crops (Laboratory + Field). Overall pass = **83.3%**, Laboratory sub-sample pass = **94.2%**, Field sub-sample pass = **72.6%**.
- **Sample 2 (94.2% Reported)**: Represented specifically the **Laboratory-only image subset**.
- **Reconciliation**: The overall end-to-end SAM2 leaf segmentation pass rate across the full mixed production dataset is **83.3%** with a **16.7% fallback rate** to Full-Image Grad-CAM.


## Severity Claim Audit
- **Ground Truth**: No ground-truth pixel-level disease masks or expert biological severity annotations exist in the dataset.
- **Scientific Wording**: Termed strictly as `"Estimated Visual Disease Coverage Proxy"`.
- **Category Bins**: Mild (<15%), Moderate (15–35%), Severe (>35%) are heuristic visual rules of thumb.


## RAG Claim Audit
- **Claim Wording Revision**: Replaced absolute claims of `"0 hallucination risk"` with:
  `"22 verified agronomic entries; no unsupported claims were detected in the evaluated RAG test suite."`


## System Latency Breakdown (NVIDIA RTX 4090)
- **Neural Network Batch Forward Pass (Model A + Model B)**: `5.93 ms` (Reported `5.12 ms` baseline)
- **CPU Image Loading & Preprocessing (PIL + PyTorch)**: `12.4 ms`
- **SAM2 On-Demand Post-Classification Segmentation**: `18.6 ms`
- **Grad-CAM Localization**: `4.8 ms`
- **RAG Agronomic Advice Retrieval**: `2.1 ms`
- **Total End-to-End Pipeline Latency**: `~43.0 ms` (P95 = `58.2 ms`)


---

## 2. SCRC Risk vs Coverage Validation Operating Points

| Policy Name | Crop Thresh | Disease Thresh | EDL Unc Thresh | Accepted | Rejected | Coverage | Selective Acc | Selective Risk |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Liberal (No Gate)** | `0.5` | `0.5` | `1.0` | 4,936 | 36 | `99.28%` | `98.46%` | `1.54%` |
| **Standard Operating Point** | `0.85` | `0.7` | `0.35` | 1,976 | 2,996 | `39.74%` | `99.04%` | `0.96%` |
| **High-Precision SCRC Gate** | `0.98` | `0.9` | `0.15` | 619 | 4,353 | `12.45%` | `99.84%` | `0.16%` |
| **Ultra-Strict SCRC Gate** | `0.99` | `0.95` | `0.08` | 37 | 4,935 | `0.74%` | `100.0%` | `0.0%` |
