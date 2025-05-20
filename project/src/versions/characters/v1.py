"""
Character detection module using YOLO.

This module provides functionality to detect characters in images
using the YOLO (You Only Look Once) object detection model.
"""

import cv2
from ultralytics import YOLO
from src.versions.classes import YOLOOutput


def main(
    img: cv2.Mat,
    yolo_model_path: str,
    conf: float = 0.7,
    iou: float = 0.7,
    **_,
) -> YOLOOutput:
    """
    Detect characters in an image using a YOLO model.

    Args:
        img (cv2.Mat): Input image for character detection.
        yolo_model_path (str): Path to the YOLO model weights.
        conf (float, optional): Confidence threshold for detections. Defaults to 0.7.
        iou (float, optional): IoU (Intersection over Union) threshold for NMS. Defaults to 0.45.

    Returns:
        YOLOOutput: Object containing detection results with the following attributes:
            - boxes: List of bounding boxes in [x1, y1, x2, y2] format.
            - class_ids: List of class IDs for each detected object.
            - confidences: List of confidence scores for each detection.
    """
    yolo_model: YOLO = YOLO(
        model=yolo_model_path,
        task="detect",
    )

    prediction = yolo_model.predict(
        source=img,
        conf=conf,
        iou=iou,
    )[0]

    if len(prediction.boxes) == 0:
        return YOLOOutput(
            boxes=[],
            class_ids=[],
            confidences=[],
        )

    boxes = prediction.boxes.xyxy
    classes_ids = prediction.boxes.cls.int().tolist()
    confidences = prediction.boxes.conf.tolist()

    return YOLOOutput(
        boxes=boxes,
        class_ids=classes_ids,
        confidences=confidences,
    )
