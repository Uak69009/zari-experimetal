"""
ZARI.ai Backend — LLM Prompt Templates
Structured prompts for bilingual (Urdu/English) crop disease advisory generation
via Groq API (Llama-3-8B).
"""


SYSTEM_PROMPT_EN = """You are ZARI, an expert agricultural advisor for Pakistani farmers. 
You specialize in crop disease diagnosis and treatment recommendations.

Your responses must be:
1. Practical and actionable for smallholder farmers in Pakistan.
2. Include both organic/traditional and chemical treatment options.
3. Mention locally available products and brands when possible.
4. Include preventive measures for future crops.
5. Be sensitive to the farmer's economic constraints.

Always structure your response with:
- Disease Name & Severity
- Immediate Treatment Steps
- Chemical Treatment (with dosage)
- Organic/Traditional Alternative
- Prevention for Next Season
"""


SYSTEM_PROMPT_UR = """آپ زری ہیں، پاکستانی کسانوں کے لیے ایک ماہر زرعی مشیر۔
آپ فصل کی بیماریوں کی تشخیص اور علاج کی سفارشات میں مہارت رکھتے ہیں۔

آپ کے جوابات:
1. پاکستان کے چھوٹے کسانوں کے لیے عملی اور قابل عمل ہونے چاہئیں۔
2. نامیاتی/روایتی اور کیمیائی دونوں علاج کے اختیارات شامل کریں۔
3. مقامی طور پر دستیاب مصنوعات اور برانڈز کا ذکر کریں۔
4. آئندہ فصلوں کے لیے احتیاطی تدابیر شامل کریں۔
5. کسان کی معاشی حدود کا خیال رکھیں۔

ہمیشہ اپنا جواب اس ترتیب سے دیں:
- بیماری کا نام اور شدت
- فوری علاج کے اقدامات
- کیمیائی علاج (خوراک کے ساتھ)
- نامیاتی/روایتی متبادل
- اگلے موسم کے لیے احتیاط
"""


def build_advisory_prompt(
    disease_label: str,
    confidence: float,
    crop_name: str,
    context_docs: list,
    language: str = "ur",
) -> dict:
    """
    Build a structured prompt for Groq LLM advisory generation.

    Args:
        disease_label: Diagnosed disease canonical label.
        confidence: Model confidence score (0-1).
        crop_name: Name of the affected crop.
        context_docs: Retrieved RAG context documents.
        language: Response language ('ur' or 'en').

    Returns:
        dict with 'system' and 'user' prompt strings.
    """
    system_prompt = SYSTEM_PROMPT_UR if language == "ur" else SYSTEM_PROMPT_EN

    # Build context string from retrieved documents
    context_str = ""
    if context_docs:
        context_str = "\n\n--- Retrieved Agricultural Knowledge ---\n"
        for i, doc in enumerate(context_docs, 1):
            context_str += f"\n[Document {i}]: {doc.get('document', '')}\n"
        context_str += "\n--- End of Retrieved Knowledge ---\n"

    # Build user prompt
    if language == "ur":
        user_prompt = f"""فصل: {crop_name}
تشخیص شدہ بیماری: {disease_label}
ماڈل کا اعتماد: {confidence * 100:.1f}%

{context_str}

براہ کرم اس بیماری کے بارے میں مکمل مشورہ دیں جس میں علاج، نامیاتی متبادل، اور احتیاطی تدابیر شامل ہوں۔
جواب اردو میں دیں۔"""
    else:
        user_prompt = f"""Crop: {crop_name}
Diagnosed Disease: {disease_label}
Model Confidence: {confidence * 100:.1f}%

{context_str}

Please provide a comprehensive advisory for this disease including treatment steps, 
organic alternatives, and preventive measures for Pakistani farming conditions.
Respond in English."""

    return {
        "system": system_prompt,
        "user": user_prompt,
    }


def build_fallback_prompt(language: str = "ur") -> str:
    """
    Return a fallback message when the model confidence is too low.

    Args:
        language: Response language.

    Returns:
        Fallback message string.
    """
    if language == "ur":
        return (
            "⚠️ تصویر واضح نہیں ہے یا بیماری کی شناخت ممکن نہیں۔\n\n"
            "براہ کرم:\n"
            "1. پتی کی قریبی اور صاف تصویر بھیجیں\n"
            "2. روشنی اچھی ہو اس کا خیال رکھیں\n"
            "3. بیمار حصے کو واضح طور پر دکھائیں\n\n"
            "یا اپنا سوال ٹیکسٹ میں لکھ کر بھیجیں۔"
        )
    else:
        return (
            "⚠️ The image is unclear or the disease could not be identified.\n\n"
            "Please:\n"
            "1. Send a close-up, clear photo of the affected leaf\n"
            "2. Ensure good lighting\n"
            "3. Show the diseased area clearly\n\n"
            "Or type your question as a text message."
        )
