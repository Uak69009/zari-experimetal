# 🌿 ZARI.ai — Zero-Cost Agricultural Intelligence

> **Multi-modal crop disease diagnosis and advisory platform for Pakistani farmers.**

ZARI.ai bridges the lab-to-field domain shift by combining state-of-the-art computer vision with retrieval-augmented generation (RAG) to deliver accurate, localized crop disease diagnoses — entirely free of inference costs.

---

## 🚀 Key Features

| Feature | Technology | Cost |
|---|---|---|
| **Image Diagnosis** | EfficientNetV2-S → ONNX (CPU) | Free |
| **Advisory Generation** | Groq API (Llama-3-8B) + ChromaDB RAG | Free |
| **Voice Input** | `faster-whisper` (Urdu / English STT) | Free |
| **Voice Output** | `edge-tts` (`ur-PK-AsadNeural`) | Free |
| **WhatsApp Interface** | Meta Business Cloud API Webhooks | Free tier |
| **Web Dashboard** | Next.js 14 PWA (mobile-first) | Self-hosted |

---

## 📋 Architecture Overview

```
User (WhatsApp / Web)
        │
        ▼
   FastAPI Gateway ──────────────────────┐
        │                                 │
        ▼                                 ▼
  ONNX CV Model              faster-whisper (STT)
  (EfficientNetV2-S)                │
        │                           ▼
        │                    Text query extracted
        │                           │
        ▼                           ▼
   Disease Label ──────►  ChromaDB RAG Retrieval
   (>85% conf.)                     │
        │                           ▼
        ▼                    Groq LLM Advisory
   Quality Gate              (Llama-3-8B)
   (85% Softmax)                    │
        │                           ▼
        ▼                    edge-tts Voice Reply
   Diagnosis + Advisory ──► WhatsApp / Web Response
```

---

## 🛠️ Installation

### Prerequisites

- Python 3.10+
- Node.js 18+
- Git

### 1. Clone the Repository

```bash
git clone https://github.com/Uak69009/ZARI.ai.git
cd ZARI.ai/zari-ai
```

### 2. Backend Setup

```bash
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Frontend Setup

```bash
cd frontend
npm install
```

### 4. ML Pipeline Setup (for training only)

```bash
cd ml_pipeline
pip install -r requirements_ml.txt
```

---

## 🔑 Environment Variables

Create a `.env` file in the `backend/` directory:

```env
# ── Groq LLM ──
GROQ_API_KEY=gsk_your_groq_api_key_here

# ── Meta WhatsApp Business Cloud API ──
WHATSAPP_VERIFY_TOKEN=your_custom_verify_token
WHATSAPP_ACCESS_TOKEN=your_meta_access_token
WHATSAPP_PHONE_NUMBER_ID=your_phone_number_id

# ── ONNX Model ──
ONNX_MODEL_PATH=models/efficientnetv2s_zari.onnx
CONFIDENCE_THRESHOLD=0.85

# ── ChromaDB ──
CHROMA_PERSIST_DIR=./chroma_db
CHROMA_COLLECTION=zari_agri_docs

# ── Voice Pipeline ──
WHISPER_MODEL_SIZE=base
TTS_VOICE=ur-PK-AsadNeural

# ── Server ──
HOST=0.0.0.0
PORT=8000
DEBUG=true
```

---

## 🧪 Running the Application

### Backend (Development)

```bash
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Frontend (Development)

```bash
cd frontend
npm run dev
```

---

## 📊 Datasets Used

| Dataset | Images | Source | Role |
|---|---|---|---|
| **PlantVillage** | 54,305 | [GitHub](https://github.com/spMohanty/PlantVillage-Dataset) | Lab baseline |
| **PlantDoc** | 2,598 | [GitHub](https://github.com/pratikkayal/PlantDoc-Dataset) | Noisy backgrounds |
| **PlantCity** | 10,667 | [Kaggle](https://www.kaggle.com/datasets/codewithsk/plantcity-a-comprehensive-images-multicrop-leaves) | Pakistani field images |
| **NWRD** | 17,856 | [GitHub](https://github.com/dll-ncai/NUST-Wheat-Rust-Disease-NWRD) | Wheat rust focus |

---

## 🎯 Target Metrics

- **Accuracy:** >95% on Pakistani field test data (PlantCity + NWRD held-out)
- **Quality Gate:** 85% Softmax confidence threshold
- **Latency:** <3 seconds end-to-end (image → advisory)
- **Cost:** $0/inference (ONNX CPU + Groq free tier + edge-tts)

---

## 👤 Author

**Umair Amjad Khan**

---

## 📄 License

This project is licensed under the MIT License.
