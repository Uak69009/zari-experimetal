"""
ZARI.ai Backend — WhatsApp Business Cloud API Webhook
Handles incoming messages from Meta WhatsApp Business Cloud API.
"""

from fastapi import APIRouter, Request, HTTPException, Query
from fastapi.responses import PlainTextResponse
import httpx
import json

from core.config import settings

router = APIRouter()


@router.get("")
async def verify_webhook(
    hub_mode: str = Query(None, alias="hub.mode"),
    hub_verify_token: str = Query(None, alias="hub.verify_token"),
    hub_challenge: str = Query(None, alias="hub.challenge"),
):
    """
    WhatsApp webhook verification endpoint.
    Meta sends a GET request with a challenge to verify the webhook URL.
    """
    if hub_mode == "subscribe" and hub_verify_token == settings.whatsapp_verify_token:
        return PlainTextResponse(content=hub_challenge, status_code=200)
    
    raise HTTPException(status_code=403, detail="Verification failed.")


@router.post("")
async def receive_message(request: Request):
    """
    Handle incoming WhatsApp messages.
    
    Routes messages based on type:
    - image → CV diagnosis pipeline
    - audio → STT → text query pipeline
    - text  → Direct RAG query pipeline
    """
    body = await request.json()

    try:
        # Extract message data from webhook payload
        entry = body.get("entry", [{}])[0]
        changes = entry.get("changes", [{}])[0]
        value = changes.get("value", {})
        messages = value.get("messages", [])

        if not messages:
            # Status update or other non-message event
            return {"status": "ok"}

        message = messages[0]
        sender_phone = message.get("from", "")
        message_type = message.get("type", "")

        if message_type == "image":
            await _handle_image_message(message, sender_phone)
        elif message_type == "audio":
            await _handle_audio_message(message, sender_phone)
        elif message_type == "text":
            await _handle_text_message(message, sender_phone)
        else:
            await _send_whatsapp_text(
                sender_phone,
                "⚠️ براہ کرم فصل کی بیمار پتی کی تصویر بھیجیں۔\n"
                "Please send a photo of the affected crop leaf.",
            )

        return {"status": "ok"}

    except Exception as e:
        print(f"Webhook processing error: {e}")
        return {"status": "error", "detail": str(e)}


async def _handle_image_message(message: dict, sender_phone: str):
    """Process image message: download → diagnose → reply."""
    media_id = message.get("image", {}).get("id", "")

    # Step 1: Download image from Meta Graph API
    image_bytes = await _download_whatsapp_media(media_id)

    if image_bytes is None:
        await _send_whatsapp_text(sender_phone, "❌ تصویر ڈاؤن لوڈ نہیں ہو سکی۔")
        return

    # Step 2: Run CV inference
    # TODO: Integrate with CropDiseaseClassifier
    # result = classifier.predict(image_bytes)

    # Step 3: Generate advisory via RAG + LLM
    # TODO: Integrate with RAG pipeline

    # Step 4: Send text + audio reply
    # TODO: Send diagnosis result
    await _send_whatsapp_text(
        sender_phone,
        "🌿 تصویر موصول ہوئی۔ تشخیص جاری ہے...\n"
        "Image received. Diagnosis in progress...",
    )


async def _handle_audio_message(message: dict, sender_phone: str):
    """Process audio message: download → transcribe → query."""
    media_id = message.get("audio", {}).get("id", "")

    # TODO: Download audio, transcribe with faster-whisper, process query
    await _send_whatsapp_text(
        sender_phone,
        "🎙️ آواز موصول ہوئی۔ عمل جاری ہے...\n"
        "Voice message received. Processing...",
    )


async def _handle_text_message(message: dict, sender_phone: str):
    """Process text message: direct RAG query."""
    text = message.get("text", {}).get("body", "")

    # TODO: Query RAG pipeline with text
    await _send_whatsapp_text(
        sender_phone,
        f"📝 آپ کا سوال موصول ہوا: {text}\n"
        f"Your query received: {text}\n\n"
        "براہ کرم فصل کی تصویر بھیجیں تاکہ تشخیص ہو سکے۔\n"
        "Please send a photo of the crop for diagnosis.",
    )


async def _download_whatsapp_media(media_id: str) -> bytes | None:
    """Download media file from Meta Graph API."""
    try:
        async with httpx.AsyncClient() as client:
            # Step 1: Get media URL
            url_response = await client.get(
                f"https://graph.facebook.com/v18.0/{media_id}",
                headers={"Authorization": f"Bearer {settings.whatsapp_access_token}"},
            )
            media_url = url_response.json().get("url", "")

            # Step 2: Download the actual media
            media_response = await client.get(
                media_url,
                headers={"Authorization": f"Bearer {settings.whatsapp_access_token}"},
            )
            return media_response.content

    except Exception as e:
        print(f"Media download error: {e}")
        return None


async def _send_whatsapp_text(phone_number: str, message: str):
    """Send a text message via WhatsApp Business Cloud API."""
    try:
        async with httpx.AsyncClient() as client:
            await client.post(
                f"https://graph.facebook.com/v18.0/{settings.whatsapp_phone_number_id}/messages",
                headers={
                    "Authorization": f"Bearer {settings.whatsapp_access_token}",
                    "Content-Type": "application/json",
                },
                json={
                    "messaging_product": "whatsapp",
                    "to": phone_number,
                    "type": "text",
                    "text": {"body": message},
                },
            )
    except Exception as e:
        print(f"WhatsApp send error: {e}")
