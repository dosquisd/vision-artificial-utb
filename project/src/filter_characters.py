"""
Image filtering module for braille character detection.

This module provides functions to filter and preprocess images
for improved braille character detection.
"""

import cv2
from skimage import filters
from skimage.morphology import disk
from src.config import settings


def filter(image: cv2.Mat, radius: float, amount: float) -> cv2.Mat:
    """
    Apply unsharp mask filtering to an image.

    This function applies an unsharp mask filter to enhance the edges
    in the image, which can help in better detection of braille characters.

    Args:
        image (cv2.Mat): Input image to be filtered (should be in BGR format)
        radius (float): Radius parameter for the unsharp mask filter
        amount (float): Amount/strength parameter for the unsharp mask filter

    Returns:
        cv2.Mat: The filtered image
    """
    # Convertir a escala de grises
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Hacer un resize
    gray = cv2.resize(gray, settings.PROCESSED_CHARACTER_SHAPE[::-1])

    # Aplicar más filtros aquí
    gray = filters.median(gray, disk(3))
    gray = filters.unsharp_mask(gray, radius=radius, amount=amount)
    gray = (gray * 255).astype("uint8")

    return gray


def main_filter(image_path: str, radius: float, amount: float) -> cv2.Mat:
    """
    Apply a series of image processing filters to prepare an image for braille character detection.

    This function loads an image from disk, converts it to grayscale, resizes it based on
    project settings, and applies additional filters including median filtering and
    unsharp masking for enhanced character detection.

    Args:
        image_path (str): Path to the image file to be processed
        radius (float): Radius parameter for the unsharp mask filter
        amount (float): Amount/strength parameter for the unsharp mask filter

    Returns:
        cv2.Mat: The processed grayscale image
    """
    image = cv2.imread(image_path)
    return filter(image, radius, amount)
