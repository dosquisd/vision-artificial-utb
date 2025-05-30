"""
Character translation module using YOLO.

This module provides functionality to translate characters in images
using the YOLO (You Only Look Once) object detection model, taking
only the prediction with highest confidence.
"""

import cv2
from ultralytics import YOLO
from src.versions.classes import YOLOClassifyOutput


def main(
    img: cv2.Mat,
    yolo_model_path: str,
    conf: float = 0.7,
    iou: float = 0.7,
    **_,
) -> YOLOClassifyOutput:
    """
    Translate a character in an image using a YOLO model, returning only the highest confidence detection.

    Args:
        img (cv2.Mat): Input image for character translation.
        yolo_model_path (str): Path to the YOLO model weights.
        conf (float, optional): Confidence threshold for detections. Defaults to 0.7.
        iou (float, optional): IoU (Intersection over Union) threshold for NMS. Defaults to 0.7.

    Returns:
        YOLOClassifyOutput: A dictionary-like object containing:
            - top1_class_id (int): The predicted class ID with highest confidence
            - top1_confidence (float): The confidence score for the top prediction
            - top5_class_ids (list[int]): List of top 5 predicted class IDs
            - top5_confidences (list[float]): Confidence scores for the top 5 predictions
    """
    yolo_model: YOLO = YOLO(model=yolo_model_path, task="detect")
    prediction = yolo_model.predict(source=img, conf=conf, iou=iou)[0]

    probs = prediction.probs
    top1_cls = probs.top1
    top1_conf = probs.top1conf.item()
    top5_cls = probs.top5
    top5_conf = probs.top5conf.tolist()

    return YOLOClassifyOutput(
        top1_class_id=top1_cls,
        top1_confidence=top1_conf,
        top5_class_ids=top5_cls,
        top5_confidences=top5_conf,
    )
