"""
ZARI.ai Backend — Text-to-Speech Service
Uses edge-tts for zero-cost Urdu/English voice synthesis.
"""

import asyncio
import tempfile
import os

from core.config import settings


# Voice mapping
VOICE_MAP = {
    "ur": "ur-PK-AsadNeural",       # Urdu (Pakistan) - Male
    "ur-f": "ur-PK-UzmaNeural",     # Urdu (Pakistan) - Female
    "en": "en-US-GuyNeural",         # English (US) - Male
    "en-f": "en-US-JennyNeural",     # English (US) - Female
}


async def synthesize_speech(
    text: str,
    language: str = "ur",
    output_format: str = "mp3",
) -> bytes:
    """
    Convert text to speech audio using edge-tts.

    Args:
        text: Text content to synthesize.
        language: Language code ('ur' for Urdu, 'en' for English).
        output_format: Output audio format ('mp3').

    Returns:
        Audio file bytes (MP3).
    """
    import edge_tts

    voice = VOICE_MAP.get(language, settings.tts_voice)

    # Create temporary output file
    with tempfile.NamedTemporaryFile(
        suffix=f".{output_format}", delete=False
    ) as tmp:
        tmp_path = tmp.name

    try:
        # Generate speech
        communicate = edge_tts.Communicate(text=text, voice=voice)
        await communicate.save(tmp_path)

        # Read generated audio
        with open(tmp_path, "rb") as f:
            audio_bytes = f.read()

        return audio_bytes

    finally:
        # Cleanup temp file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


async def synthesize_advisory_audio(
    advisory_text: str,
    language: str = "ur",
) -> bytes:
    """
    Synthesize a full crop disease advisory into audio.
    Splits long text into manageable chunks for better quality.

    Args:
        advisory_text: Full advisory text to convert.
        language: Language code.

    Returns:
        Complete audio bytes (MP3).
    """
    # For shorter advisories, synthesize directly
    if len(advisory_text) < 2000:
        return await synthesize_speech(advisory_text, language)

    # For longer text, split by paragraphs and concatenate
    paragraphs = advisory_text.split("\n\n")
    audio_chunks = []

    for paragraph in paragraphs:
        if paragraph.strip():
            chunk = await synthesize_speech(paragraph.strip(), language)
            audio_chunks.append(chunk)

    # Simple concatenation (works for MP3 streams)
    return b"".join(audio_chunks)
