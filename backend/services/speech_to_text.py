"""
ZARI.ai Backend — Speech-to-Text Service
Uses faster-whisper for Urdu/English audio transcription.
"""

import io
import tempfile
import os
from typing import Optional

from core.config import settings

# Lazy-loaded model instance
_whisper_model = None


def _get_whisper_model():
    """Lazy-load the faster-whisper model (singleton)."""
    global _whisper_model
    if _whisper_model is None:
        from faster_whisper import WhisperModel

        _whisper_model = WhisperModel(
            settings.whisper_model_size,
            device="cpu",
            compute_type="int8",
        )
    return _whisper_model


async def transcribe_audio(
    audio_bytes: bytes,
    language: Optional[str] = None,
) -> dict:
    """
    Transcribe audio bytes to text using faster-whisper.

    Args:
        audio_bytes: Raw audio file bytes (OGG, WAV, MP3).
        language: Optional language hint ('ur' for Urdu, 'en' for English).
                  If None, auto-detect language.

    Returns:
        dict with keys:
            - text (str): Transcribed text.
            - language (str): Detected or specified language code.
            - confidence (float): Average transcription confidence.
    """
    model = _get_whisper_model()

    # Write audio to temp file (faster-whisper requires file path)
    with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        segments, info = model.transcribe(
            tmp_path,
            language=language,
            beam_size=5,
            vad_filter=True,
        )

        # Collect all segments
        full_text = ""
        total_confidence = 0.0
        segment_count = 0

        for segment in segments:
            full_text += segment.text + " "
            total_confidence += segment.avg_logprob
            segment_count += 1

        avg_confidence = (
            total_confidence / segment_count if segment_count > 0 else 0.0
        )

        return {
            "text": full_text.strip(),
            "language": info.language if info else (language or "unknown"),
            "confidence": round(avg_confidence, 4),
        }

    finally:
        # Cleanup temp file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
