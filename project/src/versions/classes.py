"""
Type definitions for model inputs and outputs.

This module defines TypedDict classes for standardizing the output formats
of different detection and translation models used in the project.
"""

import numpy as np
from torch import Tensor
from typing import TypedDict, Union


class OutputPrediction(TypedDict):
    """
    Final output structure for the main processing pipeline.
    
    Attributes:
        result (np.ndarray): The annotated image showing detections
        boxes (list[list[int]]): List of bounding boxes for detected characters
        character_predicted (list[str]): List of predicted characters for each detection
    """

    result: np.ndarray
    boxes: list[list[int]]
    character_predicted: list[str]


class YOLOInput(TypedDict):
    """
    Input structure for functions that use YOLO models.

    Attributes:
        yolo_model_path (str): Path to the YOLO model weights.
        conf (float): Confidence threshold for detections.
        iou (float): IoU (Intersection over Union) threshold for NMS.
    """

    yolo_model_path: str
    conf: float
    iou: float


class YOLOOutput(TypedDict):
    """
    Output structure for functions that use YOLO models.

    Attributes:
        boxes (list[list[int]]): List of bounding boxes, each represented as [x1, y1, x2, y2].
        confidences (list[float]): List of confidence scores corresponding to each detection.
        class_ids (list[int]): List of class IDs corresponding to each detection.
    """

    boxes: list[Tensor]
    confidences: list[float]
    class_ids: list[int]


class PytorchTranslationInput(TypedDict):
    """
    Input structure for functions that use PyTorch translation models.

    Attributes:
        character (cv2.Mat): Input character image to be translated.
        translation_model_path (str): Path to the trained PyTorch model checkpoint.
        device (str): Device to run inference on ('cpu' or 'cuda').
    """

    translation_model_path: str
    device: str


class PytorchTranslationOutput(TypedDict):
    """
    Output structure for functions that use PyTorch translation models.

    Attributes:
        class_id (int): Predicted class ID for the input.
    """

    class_id: int


# Type aliases for input/output models
InputCharacterModel = Union[YOLOInput]
"""Type alias for character detection model inputs, unifying various input format types."""

OutputCharacterModel = Union[YOLOOutput]
"""Type alias for character detection model outputs, unifying various output format types."""

InputTranslationModel = Union[YOLOInput, PytorchTranslationInput]
"""Type alias for character translation model inputs, unifying various input format types."""

OutputTranslationModel = Union[YOLOOutput, PytorchTranslationOutput]
"""Type alias for character translation model outputs, unifying various output format types."""
