"""
Character translation module using an ONNX classification model.

This module loads an ONNX classification model to predict the character class
from an input image, returning the top-1 and top-5 class predictions.
"""

import cv2
import numpy as np
import onnxruntime as ort
from src.versions.classes import YOLOClassifyOutput


def preprocess_image(img: cv2.Mat, input_size: tuple[int, int]) -> np.ndarray:
    # Resize image, normalize and convert BGR to RGB
    img_resized = cv2.resize(img, input_size)
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_normalized = img_rgb.astype(np.float32) / 255.0
    img_transposed = np.transpose(img_normalized, (2, 0, 1))  # CHW
    img_batched = np.expand_dims(img_transposed, axis=0)  # 1x3xHxW
    return img_batched


def main(
    img: cv2.Mat,
    yolo_model_path: str,
    conf: float = 0.7,
    iou: float = 0.7,
    **_,
) -> YOLOClassifyOutput:
    """
    Classify an image using an ONNX YOLO model.

    Args:
        img (cv2.Mat): Input image.
        yolo_model_path (str): Path to the ONNX classification model.
        conf (float): Confidence threshold (not used here, included for API compatibility).
        iou (float): IoU threshold (not used here).

    Returns:
        YOLOClassifyOutput: Top1 and Top5 class IDs and confidence scores.
    """

    # Load ONNX model
    session = ort.InferenceSession(yolo_model_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    # Assume default YOLOv8 input size
    input_shape = (224, 224)
    input_tensor = preprocess_image(img, input_shape)

    # Run inference
    outputs = session.run(None, {input_name: input_tensor})[0]  # Shape: [1, N_CLASSES]
    probs = outputs[0]  # remove batch dim

    # Get top1 and top5 indices and scores
    top5_indices = np.argsort(probs)[::-1][:5]
    top5_scores = probs[top5_indices].tolist()

    return YOLOClassifyOutput(
        top1_class_id=int(top5_indices[0]),
        top1_confidence=float(top5_scores[0]),
        top5_class_ids=top5_indices.tolist(),
        top5_confidences=top5_scores,
    )
