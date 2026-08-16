import asyncio
import os
import edge_tts

# Determine backend root directory and configure the target audio folder
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMP_AUDIO_DIR = os.path.join(BASE_DIR, "temp_audio")

# Ensure the temporary audio directory exists on module load
os.makedirs(TEMP_AUDIO_DIR, exist_ok=True)

async def generate_audio(text: str, message_id: str) -> str:
    """
    Converts Urdu text to spoken audio utilizing the edge-tts engine.
    
    Args:
        text (str): The Urdu text string to synthesize.
        message_id (str): Unique WhatsApp message ID used for safe file naming.
        
    Returns:
        str: The absolute file path to the generated MP3 audio file.
    """
    # Force the voice to Pakistani Urdu for accurate localized pronunciation
    voice = "ur-PK-AsadNeural"
    
    # Construct dynamic file path using the message_id
    filepath = os.path.join(TEMP_AUDIO_DIR, f"{message_id}.mp3")
    
    # Instantiate the TTS communication stream
    communicate = edge_tts.Communicate(text, voice=voice)
    
    # Execute the synthesis and save the binary data to disk
    await communicate.save(filepath)
    
    return filepath
