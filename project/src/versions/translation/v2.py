"""
Character translation module using YOLO.

This module provides functionality to translate characters in images
using the YOLO (You Only Look Once) object detection model, taking
only the prediction with highest confidence.
"""

import cv2
from ultralytics import YOLO
from src.versions.classes import YOLOOutput


def main(
    img: cv2.Mat,
    yolo_model_path: str,
    conf: float = 0.0,
    iou: float = 0.7,
    **_,
) -> YOLOOutput:
    """
    Translate a character in an image using a YOLO model, returning only the highest confidence detection.

    Args:
        img (cv2.Mat): Input image for character translation.
        yolo_model_path (str): Path to the YOLO model weights.
        conf (float, optional): Confidence threshold for detections. Defaults to 0.0.
        iou (float, optional): IoU (Intersection over Union) threshold for NMS. Defaults to 0.7.

    Returns:
        YOLOOutput: Object containing detection results with the following attributes:
            - boxes: List containing the single highest confidence bounding box in [x1, y1, x2, y2] format.
            - class_ids: List containing the class ID of the highest confidence detection.
            - confidences: List containing the confidence score of the highest confidence detection.
    """
    yolo_model: YOLO = YOLO(model=yolo_model_path, task="detect")

    prediction = yolo_model.predict(source=img, conf=conf, iou=iou)[0]

    if len(prediction.boxes) == 0:
        return YOLOOutput(
            boxes=[],
            class_ids=[],
            confidences=[],
        )

    index = prediction.boxes.conf.argmax()
    box = prediction.boxes.xyxy[index]
    confidence = prediction.boxes.conf[index]
    cls = prediction.boxes.cls[index].int()

    return YOLOOutput(boxes=[box], class_ids=[cls], confidences=[confidence])
