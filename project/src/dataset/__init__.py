"""
Dataset processing module for the Braille Translator project.

This package contains modules for processing, transforming, and managing
braille character image datasets used in training and testing models.
"""

from src.dataset.augmentation import augment_dataset, augment_image
from src.dataset.each_character_processed import get_each_character_processed
from src.dataset.each_character_raw import get_each_character_raw
from src.dataset.train_test_processed import main as process_dataset

__all__ = [
    "augment_dataset",
    "augment_image",
    "get_each_character_processed",
    "get_each_character_raw",
    "process_dataset",
]
