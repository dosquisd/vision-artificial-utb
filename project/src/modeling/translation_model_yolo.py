"""
YOLO-based translation model for braille character classification.

This module provides functionality to train a YOLOv8 classification model
for recognizing and translating individual braille characters to their
corresponding text representations using the Ultralytics implementation.
"""

from typing import Any
from ultralytics import YOLO
from src.config import settings


def main(
    dataset: str = "./data/processed/kaggle/yolo",
    project: str = "./models/runs/translation",
    basemodel: str = "./models/yolo11n-cls.pt",
    **kwargs_training: Any,
) -> None:
    """
    Trains a YOLO classification model on the specified dataset.

    Args:
        dataset (str): Path to the dataset directory in YOLO format. Defaults to "./data/processed/kaggle/yolo".
        project (str): Directory where training results and model checkpoints will be saved. Defaults to "./models/runs/translation".
        basemodel (str): Path to the base YOLO model weights file. Defaults to "./models/yolo11n-cls.pt".
        **kwargs_training: Additional keyword arguments for training configuration (e.g., batch size, learning rate, dropout, workers, optimizer, device).

    Returns:
        None
    """
    model: YOLO = YOLO(
        model=basemodel,
        task="classify",
        verbose=True,
    )

    model.train(
        data=dataset,
        epochs=settings.NUM_EPOCHS,
        imgsz=settings.PROCESSED_CHARACTER_SHAPE_YOLO[::-1],
        project=project,
        **kwargs_training,
        # Examples below
        # batch=32,
        # lr0=0.0001,
        # dropout=0.2,
        # workers=10,
        # optimizer="Adam",
        # device=0,
    )
