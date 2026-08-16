"""
ZARI.ai Backend — API Routes
REST endpoints for web-based crop disease diagnosis.
"""

from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from typing import Optional
import io

from services.cv_inference import CropDiseaseClassifier
from rag.chroma_client import get_disease_context
from rag.prompt_templates import build_advisory_prompt

router = APIRouter()


@router.post("/diagnose")
async def diagnose_crop_disease(
    image: UploadFile = File(...),
    language: Optional[str] = "ur",
):
    """
    Diagnose crop disease from an uploaded leaf image.

    Args:
        image: Uploaded image file (JPEG/PNG).
        language: Response language ('ur' for Urdu, 'en' for English).

    Returns:
        JSON with disease label, confidence, advisory text, and audio URL.
    """
    # Validate file type
    if image.content_type not in ["image/jpeg", "image/png", "image/webp"]:
        raise HTTPException(
            status_code=400,
            detail="Invalid image format. Please upload JPEG, PNG, or WebP.",
        )

    # Read image bytes
    image_bytes = await image.read()

    # ── Step 1: CV Inference ──
    classifier = CropDiseaseClassifier()
    result = classifier.predict(image_bytes)

    if not result["is_confident"]:
        return JSONResponse(
            status_code=200,
            content={
                "status": "low_confidence",
                "message": "تصویر واضح نہیں ہے۔ براہ کرم ایک صاف تصویر بھیجیں۔"
                if language == "ur"
                else "The image is unclear. Please send a clearer photo of the affected leaf.",
                "confidence": result["confidence"],
                "disease_label": None,
                "advisory": None,
            },
        )

    # ── Step 2: RAG Context Retrieval ──
    # TODO: Implement after ChromaDB is populated
    # context_docs = get_disease_context(result["disease_label"])

    # ── Step 3: LLM Advisory Generation ──
    # TODO: Implement Groq API call with RAG context
    # advisory = await generate_advisory(result["disease_label"], context_docs, language)

    # ── Step 4: TTS Voice Synthesis ──
    # TODO: Generate audio advisory
    # audio_url = await synthesize_speech(advisory, language)

    return JSONResponse(
        status_code=200,
        content={
            "status": "success",
            "disease_label": result["disease_label"],
            "confidence": result["confidence"],
            "crop": result.get("crop", "unknown"),
            "advisory": None,  # TODO: Replace with LLM advisory
            "audio_url": None,  # TODO: Replace with TTS audio URL
        },
    )


@router.post("/transcribe")
async def transcribe_audio(
    audio: UploadFile = File(...),
):
    """
    Transcribe voice input (Urdu/English) to text query.

    Args:
        audio: Uploaded audio file (OGG/WAV/MP3).

    Returns:
        JSON with transcribed text and detected language.
    """
    # TODO: Implement faster-whisper transcription
    raise HTTPException(status_code=501, detail="Voice transcription not yet implemented.")


@router.get("/taxonomy")
async def get_taxonomy():
    """Return the canonical disease taxonomy for the frontend."""
    # TODO: Load and return taxonomy.json
    raise HTTPException(status_code=501, detail="Taxonomy endpoint not yet implemented.")
