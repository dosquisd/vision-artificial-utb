"""
Image filtering module for braille character detection.

This module provides functions to filter and preprocess images
for improved braille character detection.
"""

import cv2
from src.config import settings


def main_filter(image_path: str) -> cv2.Mat:
    """
    Apply a series of image processing filters to prepare an image for braille character detection.

    This function loads an image from disk, converts it to grayscale, resizes it based on
    project settings, and applies additional necessary filters for character detection.

    Args:
        image_path (str): Path to the image file to be processed

    Returns:
        cv2.Mat: The processed grayscale image
    """
    image = cv2.imread(image_path)

    # Convertir a escala de grises
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Hacer un resize
    gray = cv2.resize(gray, settings.PROCESSED_IMAGE_SHAPE)

    # Aplicar más filtros aquí

    return gray
