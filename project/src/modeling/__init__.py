"""
Modeling module for the Braille Translator project.

This package contains modules related to model training, inference,
and evaluation for braille character detection and recognition.
"""

from src.modeling.train_yolo_model import main as train_yolo_model
from src.modeling.translation_model_yolo import main as train_translation_yolo
import src.modeling.translation_model as translation_model

__all__ = [
    "train_yolo_model",
    "train_translation_yolo",
    "translation_model",
]
