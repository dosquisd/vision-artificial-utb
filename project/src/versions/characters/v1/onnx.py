"""
Character detection module using ONNX Runtime.

This module provides functionality to detect characters in images
using a YOLOv8 ONNX model.
"""

import cv2
import numpy as np
import onnxruntime as ort
from src.versions.classes import YOLODetectOutput


def preprocess(img: np.ndarray, input_shape=(640, 640)) -> np.ndarray:
    """Resize and normalize image for YOLO ONNX input."""
    img_resized = cv2.resize(img, input_shape)
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_norm = img_rgb.astype(np.float32) / 255.0
    img_transposed = np.transpose(img_norm, (2, 0, 1))  # HWC to CHW
    return np.expand_dims(img_transposed, axis=0)  # Add batch dim


def postprocess(prediction: np.ndarray, conf_thres=0.7, iou_thres=0.7):
    """Extract boxes, confidences and class IDs from raw model output."""
    boxes = []
    confidences = []
    class_ids = []

    for pred in prediction[0]:
        conf = pred[4]
        if conf < conf_thres:
            continue

        scores = pred[5:]
        class_id = int(np.argmax(scores))
        class_score = scores[class_id]
        final_conf = conf * class_score

        if final_conf < conf_thres:
            continue

        x_center, y_center, width, height = pred[:4]
        x1 = x_center - width / 2
        y1 = y_center - height / 2
        x2 = x_center + width / 2
        y2 = y_center + height / 2

        boxes.append([x1, y1, x2, y2])
        confidences.append(final_conf)
        class_ids.append(class_id)

    return boxes, class_ids, confidences


def main(
    img: cv2.Mat,
    yolo_model_path: str,
    conf: float = 0.7,
    iou: float = 0.7,
    **_,
) -> YOLODetectOutput:
    """
    Detect characters in an image using an ONNX YOLO model.

    Args:
        img (cv2.Mat): Input image.
        yolo_model_path (str): Path to the .onnx weights file.
        conf (float): Confidence threshold.
        iou (float): IoU threshold for NMS (not yet used).

    Returns:
        YOLODetectOutput: Boxes, class IDs and confidences.
    """
    # Load ONNX model
    session = ort.InferenceSession(yolo_model_path)
    input_name = session.get_inputs()[0].name

    # Preprocess
    input_tensor = preprocess(img)

    # Inference
    outputs = session.run(None, {input_name: input_tensor})
    prediction = outputs[0]  # shape: (1, N, 85) for YOLOv8

    # Postprocess
    boxes, class_ids, confidences = postprocess(prediction, conf, iou)

    return YOLODetectOutput(
        boxes=boxes,
        class_ids=class_ids,
        confidences=confidences,
    )
