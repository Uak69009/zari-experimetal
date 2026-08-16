"""
ZARI.ai Backend — ONNX Computer Vision Inference Service
Loads EfficientNetV2-S ONNX model and performs crop disease classification
with an 85% Softmax confidence Quality Gate.
"""

import numpy as np
from PIL import Image
import io
import os
from typing import Optional

from core.config import settings

# Lazy import to avoid startup crash if model not present
_ort_session = None


def _get_onnx_session():
    """Lazy-load ONNX Runtime inference session (singleton)."""
    global _ort_session
    if _ort_session is None:
        import onnxruntime as ort

        model_path = settings.onnx_model_path
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"ONNX model not found at '{model_path}'. "
                "Please export the model using ml_pipeline/scripts/export_to_onnx.py"
            )

        _ort_session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"],
        )
    return _ort_session


# ── Canonical Class Labels ──
# This list must match the order of the model's output classes.
# Updated after taxonomy_builder.py generates the final mapping.
CLASS_LABELS = [
    # Placeholder — will be populated from taxonomy.json at startup
]


def _softmax(logits: np.ndarray) -> np.ndarray:
    """Compute softmax probabilities from raw logits."""
    exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    return exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)


def _preprocess_image(image_bytes: bytes, target_size: int = 384) -> np.ndarray:
    """
    Preprocess image for EfficientNetV2-S inference.

    Args:
        image_bytes: Raw image bytes.
        target_size: Model input resolution (384 for EfficientNetV2-S).

    Returns:
        Preprocessed numpy array of shape (1, 3, H, W).
    """
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((target_size, target_size), Image.BILINEAR)

    # Convert to numpy and normalize with ImageNet stats
    img_array = np.array(image, dtype=np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_array = (img_array - mean) / std

    # HWC → CHW → NCHW
    img_array = np.transpose(img_array, (2, 0, 1))
    img_array = np.expand_dims(img_array, axis=0)

    return img_array


class CropDiseaseClassifier:
    """
    Crop disease classifier using ONNX Runtime.
    Implements the 85% Softmax confidence Quality Gate.
    """

    def __init__(self):
        self.confidence_threshold = settings.confidence_threshold

    def predict(self, image_bytes: bytes) -> dict:
        """
        Run inference on an image and return the diagnosis.

        Args:
            image_bytes: Raw image bytes (JPEG/PNG/WebP).

        Returns:
            dict with keys:
                - disease_label (str): Predicted disease name.
                - confidence (float): Softmax probability (0-1).
                - is_confident (bool): True if confidence >= threshold.
                - class_id (int): Predicted class index.
                - crop (str): Extracted crop name from label.
        """
        try:
            session = _get_onnx_session()
        except FileNotFoundError:
            # Model not yet deployed — return mock result for development
            return {
                "disease_label": "model_not_loaded",
                "confidence": 0.0,
                "is_confident": False,
                "class_id": -1,
                "crop": "unknown",
            }

        # Preprocess
        input_tensor = _preprocess_image(image_bytes)

        # Run inference
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        logits = session.run([output_name], {input_name: input_tensor})[0]

        # Compute probabilities
        probabilities = _softmax(logits[0])
        class_id = int(np.argmax(probabilities))
        confidence = float(probabilities[class_id])

        # Map to label
        if CLASS_LABELS and class_id < len(CLASS_LABELS):
            disease_label = CLASS_LABELS[class_id]
        else:
            disease_label = f"class_{class_id}"

        # Extract crop name (convention: "crop_disease" format)
        crop = disease_label.split("_")[0] if "_" in disease_label else "unknown"

        # ── Quality Gate: 85% Softmax Threshold ──
        is_confident = confidence >= self.confidence_threshold

        return {
            "disease_label": disease_label,
            "confidence": round(confidence, 4),
            "is_confident": is_confident,
            "class_id": class_id,
            "crop": crop,
        }
