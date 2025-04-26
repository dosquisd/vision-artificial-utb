"""
YOLO model training module for braille character detection.

This module provides functionality to train a YOLOv8 model for
detecting braille characters in images using the Ultralytics implementation.
"""

from ultralytics import YOLO
from src.config import settings


def main() -> None:
    """
    Train a YOLOv8 model for braille character detection.

    This function initializes a YOLOv8 model with a pre-trained weights file
    and trains it on the braille character dataset using configuration
    parameters from the settings module.

    The training process uses the Ultralytics YOLO implementation and
    saves the trained model and results in the project's model directory.
    """
    model: YOLO = YOLO(
        model="../../models/yolo11n.pt",
        task="detect",
    )

    model.train(
        data="../../data.yaml", epochs=100, imgsz=settings.PROCESSED_IMAGE_SHAPE
    )


if __name__ == "__main__":
    main()
