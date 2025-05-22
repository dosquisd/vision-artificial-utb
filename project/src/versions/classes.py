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
        orig_img (np.ndarray): The original input image.
        result (np.ndarray): The annotated image showing detections
        boxes (list[list[int]]): List of bounding boxes for detected characters
        character_predicted (list[str]): List of predicted characters for each detection
    """

    orig_img: np.ndarray
    result_img: np.ndarray
    boxes: list[list[int]]
    character_predicted: list[str]
    confidences: list[float | None]


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


class YOLODetectOutput(TypedDict):
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


class YOLOClassifyOutput(TypedDict):
    """
    Output structure for functions that use YOLO classification models.

    Attributes:
        top1_class_id (int): The predicted class ID with the highest confidence (top-1 prediction).
        top1_confidence (float): The confidence score associated with the top-1 predicted class.
        top5_class_ids (list[int]): List of the top 5 predicted class IDs, ordered by confidence (descending).
        top5_confidences (list[float]): List of confidence scores corresponding to each class ID in `top5_class_ids`.
    """

    top1_class_id: int
    top1_confidence: float
    top5_class_ids: list[int]
    top5_confidences: list[float]


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
        top1_class_id (int): The predicted class ID with the highest confidence (top-1 prediction).
        top1_confidence (float): The confidence score associated with the top-1 predicted class.
        top5_class_ids (list[int]): List of the top 5 predicted class IDs, ordered by confidence (descending).
        top5_confidences (list[float]): List of confidence scores corresponding to each class ID in `top5_class_ids`.
    """

    top1_class_id: int
    top1_confidence: float
    top5_class_ids: list[int]
    top5_confidences: list[float]


# Type aliases for input/output models
InputCharacterModel = Union[YOLOInput]
"""Type alias for character detection model inputs, unifying various input format types."""

OutputCharacterModel = Union[YOLODetectOutput]
"""Type alias for character detection model outputs, unifying various output format types."""

InputTranslationModel = Union[YOLOInput, PytorchTranslationInput]
"""Type alias for character translation model inputs, unifying various input format types."""

OutputTranslationModel = Union[YOLOClassifyOutput, PytorchTranslationOutput]
"""Type alias for character translation model outputs, unifying various output format types."""
