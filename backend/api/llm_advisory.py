import os
import json
from groq import Groq

# Initialize the Groq client
# This expects the GROQ_API_KEY environment variable to be set on the server.
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

def generate_advisory(taxonomy_data: dict) -> str:
    """
    Synthesizes the classification data into a natural Urdu response using Groq.
    The response is tailored for conversational audio playback via Edge-TTS.
    """
    # System prompt establishing ZARI's persona
    system_prompt = (
        "You are 'ZARI', an expert Pakistani agricultural advisor. "
        "Your goal is to help local farmers by providing clear, empathetic, and highly actionable advice "
        "based on the crop disease data provided.\n\n"
        "CRITICAL INSTRUCTIONS:\n"
        "1. Write your response strictly in Urdu script.\n"
        "2. Keep the response highly conversational, concise, and easy to understand.\n"
        "3. This text will be processed by a text-to-speech (TTS) engine, so DO NOT use markdown, bullet points, asterisks, or complex symbols. Use natural punctuation.\n"
        "4. Start with a warm greeting, state the identified crop and disease, and provide immediate actionable treatment steps."
    )
    
    # Pass the taxonomy JSON block as context
    user_prompt = (
        f"Please provide an advisory response based on the following classification data:\n"
        f"{json.dumps(taxonomy_data, indent=2, ensure_ascii=False)}"
    )

    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                }
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.6,
            max_tokens=300
        )
        
        return chat_completion.choices[0].message.content

    except Exception as e:
        print(f"Groq API Error generating advisory: {e}")
        # Return a polite, default Urdu fallback message if the LLM fails
        return "معذرت، میں ابھی جواب دینے سے قاصر ہوں۔"
