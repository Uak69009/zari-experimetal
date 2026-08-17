# ZARI.ai — End-to-End Production System Architecture

## System Overview
ZARI.ai is a hierarchical, uncertainty-aware precision agricultural disease classification framework powered by PyTorch, EfficientNetV2-B2, Evidential Deep Learning (EDL), Selective Classification Risk Control (SCRC), Segment Anything Model 2 (SAM2), and Source-Grounded Retrieval-Augmented Generation (RAG).

## Components & Modules
1. **Hierarchical Model A Crop Router**: 3-class classifier isolating Tomato, Potato, and Pepper.
2. **Model B Disease Classifiers**: 3 independent crop-specific Evidential Deep Learning models parameterizing Dirichlet distributions $	ext{Dir}(oldsymbol{lpha})$.
3. **EDL & SCRC Uncertainty Gate**: Computes epistemic uncertainty $u = K / S$, enforcing a $2.0\%$ target selective risk policy.
4. **SAM2 Post-Classification Leaf Segmentation**: Runs on-demand after classification acceptance to isolate leaf geometry.
5. **Grad-CAM Visual Localization**: Extracts feature activation maps from backbone layer `features.7.1`.
6. **Visual Coverage Proxy**: Measures estimated visual disease coverage percentage and assigns heuristic severity categories.
7. **RAG Agronomic Knowledge Base**: Retrieves source-grounded diagnostic guidance and management recommendations.
