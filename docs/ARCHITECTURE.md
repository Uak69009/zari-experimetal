# 🏗️ ZARI.ai — System Architecture

> This document details the end-to-end asynchronous flow of ZARI.ai, from user input to diagnosis delivery.

---

## 1. High-Level System Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          CLIENT LAYER                                   │
│                                                                         │
│   ┌──────────────┐              ┌──────────────────────┐                │
│   │  WhatsApp     │              │  Next.js PWA          │               │
│   │  (Business    │              │  (Mobile-first        │               │
│   │   Cloud API)  │              │   Web Dashboard)      │               │
│   └──────┬───────┘              └──────────┬───────────┘                │
│          │                                  │                            │
└──────────┼──────────────────────────────────┼────────────────────────────┘
           │   Webhook POST                   │   REST API
           ▼                                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        GATEWAY LAYER (FastAPI)                          │
│                                                                         │
│   ┌──────────────────┐    ┌──────────────────┐                          │
│   │  /webhook         │    │  /api/diagnose    │                         │
│   │  (WhatsApp        │    │  (Web upload      │                         │
│   │   verification    │    │   endpoint)       │                         │
│   │   + message       │    │                   │                         │
│   │   routing)        │    │                   │                         │
│   └────────┬─────────┘    └────────┬─────────┘                          │
│            │                        │                                    │
│            └────────┬───────────────┘                                    │
│                     ▼                                                    │
│            ┌────────────────┐                                            │
│            │  Input Router   │──── Determines input type:                │
│            │                 │     IMAGE → CV Pipeline                   │
│            │                 │     AUDIO → STT Pipeline                  │
│            │                 │     TEXT  → RAG Pipeline                  │
│            └────────┬───────┘                                            │
│                     │                                                    │
└─────────────────────┼────────────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      PROCESSING LAYER                                   │
│                                                                         │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │                    VISION PIPELINE                               │   │
│   │                                                                   │   │
│   │   1. Image Preprocessing (resize, normalize)                     │   │
│   │   2. ONNX Runtime Inference (EfficientNetV2-S)                   │   │
│   │   3. Softmax Confidence Extraction                               │   │
│   │   4. Quality Gate (≥85% → proceed, <85% → fallback)             │   │
│   │                                                                   │   │
│   └──────────────────────────┬──────────────────────────────────────┘   │
│                               │                                          │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │                    VOICE PIPELINE                                │   │
│   │                                                                   │   │
│   │   1. Audio decoding (OGG/WAV)                                    │   │
│   │   2. faster-whisper STT (Urdu / English)                         │   │
│   │   3. Text query extraction                                       │   │
│   │                                                                   │   │
│   └──────────────────────────┬──────────────────────────────────────┘   │
│                               │                                          │
│                               ▼                                          │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │                 RAG RETRIEVAL PIPELINE                            │   │
│   │                                                                   │   │
│   │   1. Build query from disease label + user text                  │   │
│   │   2. ChromaDB similarity search (top-k=5)                       │   │
│   │   3. Retrieve relevant agricultural documents                    │   │
│   │   4. Construct prompt with context + disease label               │   │
│   │                                                                   │   │
│   └──────────────────────────┬──────────────────────────────────────┘   │
│                               │                                          │
│                               ▼                                          │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │                 LLM ADVISORY GENERATION                          │   │
│   │                                                                   │   │
│   │   1. Groq API call (Llama-3-8B, ~800 token response)            │   │
│   │   2. Structured output: disease name, severity, treatment,      │   │
│   │      organic alternatives, prevention                            │   │
│   │   3. Bilingual output (Urdu + English)                           │   │
│   │                                                                   │   │
│   └──────────────────────────┬──────────────────────────────────────┘   │
│                               │                                          │
│                               ▼                                          │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │                 TTS VOICE SYNTHESIS                               │   │
│   │                                                                   │   │
│   │   1. edge-tts synthesis (ur-PK-AsadNeural)                       │   │
│   │   2. MP3 audio buffer generation                                 │   │
│   │   3. Return audio alongside text advisory                       │   │
│   │                                                                   │   │
│   └──────────────────────────┬──────────────────────────────────────┘   │
│                               │                                          │
└───────────────────────────────┼──────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      RESPONSE DELIVERY                                  │
│                                                                         │
│   WhatsApp ← Send text + audio via Meta Cloud API                      │
│   Web      ← Return JSON + audio blob via REST                        │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Detailed Asynchronous Flow: WhatsApp Image Diagnosis

This is the primary user journey — a farmer sends a photo of a diseased leaf via WhatsApp.

### Step-by-Step Flow

```
 Time ──────────────────────────────────────────────────────────────────►

 T₀   │ Farmer sends leaf image via WhatsApp
      │
 T₁   │ Meta Cloud API sends POST /webhook to FastAPI
      │   → Payload contains: phone number, media_id, message type
      │
 T₂   │ FastAPI webhook handler:
      │   → Validates HMAC signature
      │   → Extracts media_id
      │   → Downloads image via Meta Graph API (httpx async)
      │
 T₃   │ Input Router identifies IMAGE type
      │   → Dispatches to cv_inference.py
      │
 T₄   │ CV Inference Pipeline:
      │   → PIL.Image resize to 384×384
      │   → Normalize (ImageNet mean/std)
      │   → ONNX Runtime session.run()
      │   → Extract class probabilities via Softmax
      │
 T₅   │ Quality Gate Decision:
      │   ├── confidence ≥ 0.85 → disease_label + confidence score
      │   └── confidence < 0.85 → "low_confidence" flag
      │
 T₆   │ RAG Context Retrieval (ChromaDB):
      │   → Query: f"{disease_label} treatment Pakistan {crop_name}"
      │   → Retrieve top-5 relevant document chunks
      │   → Construct LLM prompt with disease context
      │
 T₇   │ LLM Advisory Generation (Groq API):
      │   → Model: llama3-8b-8192
      │   → System prompt: agricultural expert + bilingual
      │   → Returns structured advisory (Urdu + English)
      │
 T₈   │ TTS Voice Synthesis (edge-tts):
      │   → Convert Urdu advisory text to speech
      │   → Voice: ur-PK-AsadNeural
      │   → Output: MP3 audio buffer
      │
 T₉   │ Response Assembly:
      │   → Text message: disease name + advisory
      │   → Audio message: voice advisory
      │
 T₁₀  │ Meta Cloud API:
      │   → Send text reply to farmer's WhatsApp
      │   → Upload + send audio reply
      │
 T₁₁  │ Farmer receives diagnosis + voice advisory (~3 sec total)
```

---

## 3. Component Responsibilities

### 3.1 FastAPI Gateway (`backend/main.py`)

- Stateless ASGI application
- CORS middleware for web frontend
- Health check endpoint (`/health`)
- Routes delegation to `api/routes.py`

### 3.2 WhatsApp Webhook (`backend/api/whatsapp_webhook.py`)

- `GET /webhook` — Meta verification challenge
- `POST /webhook` — Incoming message handler
- Media download via Meta Graph API
- Message type routing (image / audio / text)

### 3.3 CV Inference Service (`backend/services/cv_inference.py`)

- Loads PyTorch JIT (`best_model_jit.pt`) once at startup (singleton)
- Preprocesses images to model input spec (224x224 RGB)
- Runs inference natively via `torch.jit` (No C++ ONNX dependency required)
- Applies 85% Softmax confidence threshold
- Returns `(disease_label, confidence, is_confident)` tuple

### 3.4 RAG Pipeline (`backend/rag/`)

- **ChromaDB Client** — Persistent vector store of agricultural documents
- **Prompt Templates** — Structured prompts for bilingual advisory generation
- Retrieves contextually relevant treatment information

### 3.5 Voice Pipeline (`backend/services/`)

- **STT** (`speech_to_text.py`) — `faster-whisper` for Urdu/English transcription
- **TTS** (`text_to_speech.py`) — `edge-tts` with `ur-PK-AsadNeural` voice

### 3.6 Frontend (`frontend/`)

- Next.js 14 App Router
- Mobile-first Progressive Web App
- Tailwind CSS with custom ZARI design tokens
- Image upload + camera capture
- Real-time diagnosis display with audio playback

---

## 4. Deployment Topology

```
┌────────────────────────────┐
│       Vercel / VPS         │
│   ┌────────────────────┐   │
│   │   Next.js Frontend  │   │
│   │   (Static + SSR)    │   │
│   └────────────────────┘   │
└────────────┬───────────────┘
             │ API calls
             ▼
┌────────────────────────────┐
│     Railway / VPS          │
│   ┌────────────────────┐   │
│   │   FastAPI Backend    │   │
│   │   + ONNX Runtime     │   │
│   │   + ChromaDB         │   │
│   │   + faster-whisper   │   │
│   └────────────────────┘   │
└────────────────────────────┘
             │
             ▼
┌────────────────────────────┐
│   External APIs (Free)     │
│   • Groq (LLM inference)  │
│   • Meta Cloud API (WA)   │
│   • edge-tts (Microsoft)  │
└────────────────────────────┘
```

---

## 5. Key Design Decisions

| Decision | Rationale |
|---|---|
| **ONNX over PyTorch Serving** | 2-4× faster CPU inference, no CUDA dependency, portable |
| **85% Confidence Threshold** | Prevents hallucinated diagnoses; triggers explicit fallback |
| **Groq over OpenAI** | Free tier, ultra-low latency (~200ms), sufficient quality |
| **edge-tts over Google TTS** | Zero cost, native Urdu voice, no API key required |
| **ChromaDB over Pinecone** | Self-hosted, no vendor lock-in, persistent local storage |
| **Two-Phase Training** | Lab pre-train → field fine-tune solves domain shift |

---

*Author: Umair Amjad Khan*
