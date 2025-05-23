"""
Character detection module versions.

This module imports and provides access to different versions of 
braille character detection implementations.
"""

from src.versions.characters.v1.yolo import main as characterv1_yolo
from src.versions.characters.v1.onnx import main as characterv1_onnx


__all__ = [
    "characterv1_yolo",
    "characterv1_onnx",
]
