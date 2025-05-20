"""
Utility functions for model validation and processing.

This module provides helper functions for checking paths, validating models,
and other utility operations used across the project.
"""

from pathlib import Path


def is_yolo_model_path(model_path: str) -> bool:
    """
    Check if a path is a valid YOLO model path.
    
    Validates that the given path exists, is located in a 'weights' directory,
    and has a .pt extension, which are typical characteristics of YOLO model files.
    
    Args:
        model_path (str): Path to check for YOLO model validity
        
    Returns:
        bool: True if the path appears to be a valid YOLO model path, False otherwise
    """
    path = Path(model_path)

    if not path.exists():
        return False
    
    if path.parent.name != "weights":
        return False
    
    if not model_path.endswith(".pt"):
        return False

    # Add more validations here
    # ...

    return True
