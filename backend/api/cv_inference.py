import os
import io
import json
import warnings
import numpy as np
from PIL import Image
import onnxruntime as rt

# --- Path Configurations ---
# Resolve the project root assuming this file is located at `backend/api/cv_inference.py`
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ONNX_MODEL_PATH = os.path.join(BASE_DIR, "ml_pipeline", "saved_models", "zari_model.onnx")
TAXONOMY_PATH = os.path.join(BASE_DIR, "ml_pipeline", "data", "taxonomy.json")

# --- Singleton Initialization ---
# 1. Load Taxonomy Mapping
def _load_taxonomy():
    if not os.path.exists(TAXONOMY_PATH):
        warnings.warn(f"Taxonomy JSON not found at {TAXONOMY_PATH}")
        return {}
    with open(TAXONOMY_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data.get("classes", {})

ID_TO_CLASS_MAP = _load_taxonomy()

# 2. Load ONNX Session globally so it loads only once upon server start
try:
    if os.path.exists(ONNX_MODEL_PATH):
        session = rt.InferenceSession(ONNX_MODEL_PATH)
    else:
        warnings.warn(f"ONNX model not found at {ONNX_MODEL_PATH}. Prediction will fail.")
        session = None
except Exception as e:
    warnings.warn(f"Failed to load ONNX model: {e}")
    session = None


# --- Core Functions ---
def image_to_tensor(image_bytes: bytes) -> np.ndarray:
    """Preprocesses a raw image byte string into an ONNX-ready numpy tensor."""
    # Open image and ensure standard RGB format
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    
    # Resize to match EfficientNetV2-S input size
    image = image.resize((224, 224))
    
    # Convert to numpy float array and scale pixel values to [0, 1]
    img_array = np.array(image, dtype=np.float32) / 255.0
    
    # Apply standard ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_array = (img_array - mean) / std
    
    # Transpose dimensions from (Height, Width, Channels) -> (Channels, Height, Width)
    img_array = np.transpose(img_array, (2, 0, 1))
    
    # Add the batch dimension: Shape becomes (1, 3, 224, 224)
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

def softmax(x: np.ndarray) -> np.ndarray:
    """Compute numerical softmax values for confidence probabilities."""
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)

def predict(image_bytes: bytes) -> dict:
    """
    Runs computer vision inference on the WhatsApp image bytes.
    Validates against the 85% confidence quality gate.
    """
    if session is None:
        return {"status": "error", "message": "Model session is not active."}
        
    try:
        # 1. Preprocess
        input_tensor = image_to_tensor(image_bytes)
        
        # 2. Run ONNX Session Inference
        # Retrieve the dynamic input name required by the compiled model
        input_name = session.get_inputs()[0].name
        outputs = session.run(None, {input_name: input_tensor})
        
        # Extract logits from the batch
        logits = outputs[0]
        
        # 3. Calculate Probabilities (Quality Gate)
        probabilities = softmax(logits)[0]
        predicted_class_id = int(np.argmax(probabilities))
        confidence = float(probabilities[predicted_class_id])
        
        # 4. Enforce 85% Softmax Threshold
        if confidence < 0.85:
            return {
                "status": "low_confidence",
                "message": "Low Confidence / Needs Manual Review",
                "confidence": confidence
            }
            
        # 5. Return success taxonomy payload
        class_id_str = str(predicted_class_id)
        if class_id_str in ID_TO_CLASS_MAP:
            class_info = ID_TO_CLASS_MAP[class_id_str]
            return {
                "status": "success",
                "confidence": confidence,
                "class_id": predicted_class_id,
                "data": class_info
            }
        else:
            return {
                "status": "error",
                "message": f"Class ID {predicted_class_id} missing from taxonomy.json."
            }
            
    except Exception as e:
        return {"status": "error", "message": f"Inference processing failed: {str(e)}"}
