# ZARI.ai - Agricultural Intelligence Platform

ZARI.ai is a multi-modal agricultural crop disease diagnosis and advisory application designed to bridge the lab-to-field domain shift for Pakistani farmers. It provides an ultra-fast, highly scalable, and zero-cost inference platform directly accessible via a Web Dashboard or WhatsApp.

## 🌿 Architecture Overview

ZARI.ai is composed of three decoupled ecosystems:

1. **Machine Learning Pipeline (`ml_pipeline/`)**
   - **Data Ingestion:** Automatically crawls, downloads, and sanitizes 4 disjoint agricultural datasets (PlantVillage, PlantDoc, PlantCity, NWRD).
   - **Canonical Taxonomy:** Maps disjoint and overlapping class labels into a unified continuous integer index (`taxonomy.json`).
   - **Model Training:** Utilizes transfer learning on a PyTorch `EfficientNetV2-S` architecture.
   - **ONNX Export:** Serializes the trained PyTorch weights into a lightweight, dynamic-batching `.onnx` binary for pure-CPU inference.

2. **FastAPI Backend & WhatsApp Webhooks (`backend/`)**
   - **Web Inference:** Exposes a `POST /predict` endpoint for web application integration.
   - **Meta WhatsApp Cloud API:** Integrates securely with Meta webhooks to receive field images from farmers.
   - **Computer Vision (CV):** Executes pure-NumPy pre-processing and ONNX inference directly in-memory, governed by a strictly enforced 85% Softmax confidence quality gate.
   - **LLM Advisory (RAG/Groq):** Injects taxonomy predictions into a custom LLM prompt utilizing the ultra-fast `llama-3.3-70b-versatile` model to synthesize actionable, conversational Urdu advisory.
   - **Voice Synthesis (Edge-TTS):** Generates localized (`ur-PK-AsadNeural`) Urdu voice notes dynamically and streams them back to the farmer via the Meta API.

3. **Next.js Web Dashboard (`frontend/`)**
   - **Framework:** Next.js (App Router), React 18, Tailwind CSS.
   - **Interactive UI:** Provides a "Live Web Diagnostics" tab enabling drag-and-drop crop analysis.
   - **Aesthetics:** Nature-inspired UI utilizing a custom `royal-green`, `leaf-green`, and `off-white` palette.
   - **System Status:** Real-time health monitoring of the FastAPI backend latency.

## 🚀 Local Setup & Installation

### 1. Backend API Configuration
Ensure you have Python 3.10+ installed.
```bash
# Navigate to project root
cd zari-ai

# Install Python Dependencies
pip install -r backend/requirements.txt -r ml_pipeline/requirements_ml.txt

# Start the FastAPI Server (Port 8000)
python backend/main.py
```

### 2. Frontend Configuration
Ensure you have Node.js and npm installed.
```bash
# Navigate to the frontend workspace
cd frontend

# Install Node Dependencies
npm install

# Start the Next.js Development Server (Port 3000)
npm run dev
```

### 3. Environment Variables
You must configure the following keys in your environment or `.env` file for the backend to function:
- `GROQ_API_KEY`: Groq Cloud API Key for Llama-3.3 LLM access.
- `WHATSAPP_VERIFY_TOKEN`: Your custom secure string for Meta Webhook verification.
- `WHATSAPP_ACCESS_TOKEN`: Meta Graph API authorization bearer token.
- `WHATSAPP_PHONE_NUMBER_ID`: The registered WhatsApp Business Phone ID for outgoing messages.

## 🧠 The Domain-Shift Strategy
ZARI.ai combats the "Lab-to-Field" domain shift by executing a two-phase training strategy:
1. **Phase 1 (Base Knowledge):** Initial pre-training on controlled, white-background lab data (`PlantVillage`, `PlantDoc`).
2. **Phase 2 (Real-World Fine Tuning):** Feature extraction freezing and targeted classifier fine-tuning using noisy, in-field datasets (`NWRD`, `PlantCity`).

## 📜 License
MIT License. Created by Umair Amjad Khan.
