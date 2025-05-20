"""
Module version definitions.

This module provides a mapping of available model versions for both
character detection and translation components of the system.
"""

from src.versions.characters import v1 as characterv1
from src.versions.translation import v1 as translationv1, v2 as translationv2

VERSIONS: dict[str, dict[str, callable]] = {
    "translation": {
        "v1": translationv1.main,
        "v2": translationv2.main,
    },
    "characters": {
        "v1": characterv1.main,
    },
}

__all__ = ["VERSIONS"]