"""
Version management module for the Braille Translator project.

This module manages different versions of the translation and character detection
pipelines, allowing easy switching between implementation versions.
It provides a registry of available pipeline versions through the VERSIONS dictionary.
"""

import src.versions.classes as classes
import src.versions.utils as utils

import importlib
from typing import Any, Dict
from dataclasses import dataclass


@dataclass
class ModelConfig:
    """
    Configuration for a model version.

    Attributes:
        module_path (str): Path to the module containing the model.
        function_name (str): Name of the function to call in the module.
        shape (tuple): Input shape for the model.
        color_space (str): Color space of the input image. Default is "RGB".
        additional_params (dict): Additional parameters for the model function.
    """

    module_path: str
    function_name: str
    shape: tuple
    color_space: str = "RGB"
    additional_params: Dict[str, Any] = None


class VersionManager:
    """Manages different versions of models and their configurations."""

    def __init__(self):
        self._versions = self._load_versions()

    def _load_versions(self) -> Dict[str, Dict[str, ModelConfig]]:
        """Load versions from the specified modules."""
        versions = {
            "translation": {
                "v1": ModelConfig(
                    module_path="src.versions.translation",
                    function_name="translationv1",
                    shape=(28, 28),
                    color_space="GRAY",
                ),
                "v2": ModelConfig(
                    module_path="src.versions.translation",
                    function_name="translationv2",
                    shape=(32, 32),
                    color_space="RGB",
                ),
            },
            "characters": {
                "v1": ModelConfig(
                    module_path="src.versions.characters",
                    function_name="characterv1",
                    shape=(640, 640),
                    color_space="RGB",
                ),
            },
        }
        return versions

    def get_model(
        self, model_type: str, version: str, format: classes.ModelFormats = "yolo"
    ) -> callable:
        """Dynamically load and return a model instance."""
        config = self._versions[model_type][version]
        module = importlib.import_module(config.module_path)
        function_name = f"{config.function_name}_{format}"

        try:
            model_func = getattr(module, function_name)
        except Exception:
            model_func = getattr(module, config.function_name)

        return model_func

    def get_config(self, model_type: str, version: str, **_) -> ModelConfig:
        """Get the configuration for a specific model version."""
        return self._versions[model_type][version]
    
    def get_extention(self, format: classes.ModelFormats) -> str:
        """Get the file extension for a specific model format."""
        if format == "onnx":
            return ".onnx"
        elif format == "yolo":
            return ".pt"
        else:
            raise ValueError(f"Unsupported format: {format}")

    def list_versions(self, model_type: str) -> list:
        """List all available versions for a specific model type."""
        return list(self._versions[model_type].keys())


version_manager = VersionManager()


__all__ = [
    "classes",
    "utils",
    "version_manager",
    "ModelConfig",
]
