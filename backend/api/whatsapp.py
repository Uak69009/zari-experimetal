import os
import httpx
import traceback
from fastapi import APIRouter, Request, Query, HTTPException, BackgroundTasks
from fastapi.responses import PlainTextResponse

# Import custom core intelligence modules
from api import cv_inference, llm_advisory, tts_engine

router = APIRouter(prefix="/whatsapp")

# Environment configurations for Meta API
VERIFY_TOKEN = os.getenv("WHATSAPP_VERIFY_TOKEN", "zari_secret_token")
WHATSAPP_ACCESS_TOKEN = os.getenv("WHATSAPP_ACCESS_TOKEN", "")
PHONE_NUMBER_ID = os.getenv("WHATSAPP_PHONE_NUMBER_ID", "")
GRAPH_API_VERSION = "v17.0"

@router.get("/webhook")
async def verify_webhook(
    hub_mode: str = Query(None, alias="hub.mode"),
    hub_verify_token: str = Query(None, alias="hub.verify_token"),
    hub_challenge: str = Query(None, alias="hub.challenge")
):
    """Meta webhook handshake verification endpoint."""
    if hub_mode == "subscribe" and hub_verify_token == VERIFY_TOKEN:
        print("Webhook verified successfully!")
        return PlainTextResponse(content=hub_challenge)
    
    raise HTTPException(status_code=403, detail="Verification failed")

async def process_whatsapp_pipeline(sender_id: str, message_id: str, media_id: str):
    """
    End-to-End autonomous pipeline:
    Meta Image Download -> CV Inference -> LLM Advisory -> TTS Synthesis -> Meta Audio Upload
    """
    filepath = None
    try:
        async with httpx.AsyncClient() as client:
            headers = {"Authorization": f"Bearer {WHATSAPP_ACCESS_TOKEN}"}
            
            # 1. Fetch Media URL from Meta
            media_info_url = f"https://graph.facebook.com/{GRAPH_API_VERSION}/{media_id}"
            res_info = await client.get(media_info_url, headers=headers)
            res_info.raise_for_status()
            media_url = res_info.json().get("url")
            
            # 2. Download Raw Image Bytes
            res_img = await client.get(media_url, headers=headers)
            res_img.raise_for_status()
            image_bytes = res_img.content
            
            # 3. Computer Vision Inference (ONNX)
            cv_result = cv_inference.predict(image_bytes)
            
            # 4. LLM Advisory Synthesis (Groq Llama-3.3)
            advisory_text = llm_advisory.generate_advisory(cv_result)
            
            # 5. Text-to-Speech Audio Generation (Edge-TTS)
            filepath = await tts_engine.generate_audio(advisory_text, message_id)
            
            # 6. Upload Audio File to Meta Media API
            upload_url = f"https://graph.facebook.com/{GRAPH_API_VERSION}/{PHONE_NUMBER_ID}/media"
            with open(filepath, "rb") as f:
                files = {"file": (f"{message_id}.mp3", f, "audio/mpeg")}
                data = {"messaging_product": "whatsapp"}
                res_upload = await client.post(upload_url, headers=headers, data=data, files=files)
                res_upload.raise_for_status()
                uploaded_media_id = res_upload.json().get("id")
            
            # 7. Send the Audio Message to the Farmer
            send_msg_url = f"https://graph.facebook.com/{GRAPH_API_VERSION}/{PHONE_NUMBER_ID}/messages"
            payload = {
                "messaging_product": "whatsapp",
                "to": sender_id,
                "type": "audio",
                "audio": {"id": uploaded_media_id}
            }
            res_send = await client.post(send_msg_url, headers=headers, json=payload)
            res_send.raise_for_status()
            print(f"Successfully delivered advisory audio to {sender_id}")
            
    except Exception as e:
        print(f"Pipeline Error for message {message_id}: {e}")
        traceback.print_exc()
    finally:
        # 8. Cleanup Temporary Audio File to free storage
        if filepath and os.path.exists(filepath):
            try:
                os.remove(filepath)
            except Exception as e:
                print(f"Failed to delete temporary audio file {filepath}: {e}")

@router.post("/webhook")
async def receive_message(request: Request, background_tasks: BackgroundTasks):
    """
    Endpoint to receive incoming WhatsApp messages.
    Dispatches media to the autonomous pipeline securely in the background.
    """
    payload = await request.json()
    
    try:
        # Safely extract Meta JSON payload
        entry = payload.get("entry", [])[0]
        changes = entry.get("changes", [])[0]
        value = changes.get("value", {})
        
        if "messages" in value:
            message = value["messages"][0]
            sender_id = message.get("from")
            message_id = message.get("id")
            msg_type = message.get("type")
            
            # Intercept image payloads
            if msg_type == "image":
                media_id = message.get("image", {}).get("id")
                if media_id:
                    print(f"Incoming image from {sender_id}. Initializing ZARI pipeline...")
                    # Hand off the heavy lifting to FastAPI's BackgroundTasks 
                    # ensuring an immediate 200 OK response is sent to Meta
                    background_tasks.add_task(process_whatsapp_pipeline, sender_id, message_id, media_id)
            else:
                print(f"Received {msg_type} message from {sender_id}. ZARI currently only processes image analysis.")
                
    except (IndexError, KeyError):
        pass
    except Exception as e:
        print(f"Critical error parsing webhook payload: {e}")
        
    # Standard 200 OK status to prevent Meta from endlessly retrying webhook deliveries
    return {"status": "success"}
