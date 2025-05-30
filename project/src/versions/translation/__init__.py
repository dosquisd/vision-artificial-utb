"""
Translation module versions.

This module imports and provides access to different versions of
braille to text translation implementations.
"""

from src.versions.translation.v1.main import main as translationv1
from src.versions.translation.v2 import main as translationv2


__all__ = [
    "translationv1",
    "translationv2",
]
