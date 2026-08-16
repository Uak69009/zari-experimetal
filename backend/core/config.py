"""
ZARI.ai Backend — Configuration Module
Loads environment variables and defines application-wide settings.
"""

from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional


class Settings(BaseSettings):
    """Application settings loaded from environment variables or .env file."""

    # ── Server ──
    host: str = Field(default="0.0.0.0", alias="HOST")
    port: int = Field(default=8000, alias="PORT")
    debug: bool = Field(default=False, alias="DEBUG")

    # ── Groq LLM ──
    groq_api_key: str = Field(default="", alias="GROQ_API_KEY")
    groq_model: str = Field(default="llama3-8b-8192", alias="GROQ_MODEL")

    # ── WhatsApp Business Cloud API ──
    whatsapp_verify_token: str = Field(default="", alias="WHATSAPP_VERIFY_TOKEN")
    whatsapp_access_token: str = Field(default="", alias="WHATSAPP_ACCESS_TOKEN")
    whatsapp_phone_number_id: str = Field(default="", alias="WHATSAPP_PHONE_NUMBER_ID")

    # ── ONNX Model ──
    onnx_model_path: str = Field(default="models/efficientnetv2s_zari.onnx", alias="ONNX_MODEL_PATH")
    confidence_threshold: float = Field(default=0.85, alias="CONFIDENCE_THRESHOLD")

    # ── ChromaDB ──
    chroma_persist_dir: str = Field(default="./chroma_db", alias="CHROMA_PERSIST_DIR")
    chroma_collection: str = Field(default="zari_agri_docs", alias="CHROMA_COLLECTION")

    # ── Voice Pipeline ──
    whisper_model_size: str = Field(default="base", alias="WHISPER_MODEL_SIZE")
    tts_voice: str = Field(default="ur-PK-AsadNeural", alias="TTS_VOICE")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


# Singleton settings instance
settings = Settings()
