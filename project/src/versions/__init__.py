from src.versions.characters import characterv1
from src.versions.translation import translationv1, translationv2
import src.versions.classes as classes
import src.versions.utils as utils

VERSIONS: dict[str, dict[str, callable]] = {
    "translation": {
        "v1": translationv1,
        "v2": translationv2,
    },
    "characters": {
        "v1": characterv1,
    },
}


__all__ = [
    "characterv1",
    "translationv1",
    "translationv2",
    "VERSIONS",
    "classes",
    "utils"
]
