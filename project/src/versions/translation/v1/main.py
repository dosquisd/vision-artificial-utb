"""
Character translation module using a PyTorch model.

This module provides functionality to translate detected characters
into their corresponding labels using a pre-trained PyTorch model.
"""

import cv2
import torch
from copy import deepcopy
from src.versions.translation.v1 import model as translation_model
from src.versions.classes import PytorchTranslationOutput


net = deepcopy(translation_model.net)
optimizer = deepcopy(translation_model.optimizer)
transform = deepcopy(translation_model.transform)
net.eval()


def main(
    img: cv2.Mat,
    translation_model_path: str,
    device: str = "cpu",
    **_,
) -> PytorchTranslationOutput:
    """
    Translate a character image into its corresponding label using a PyTorch model.

    Args:
        img (cv2.Mat): Input character image to be translated.
        translation_model_path (str): Path to the trained PyTorch model checkpoint.
        device (str, optional): Device to run inference on ('cpu' or 'cuda'). Defaults to "cpu".

    Returns:
        PytorchTranslationOutput: Object containing translation result with the following attributes:
        top1_class_id (int): The predicted class ID with the highest confidence (top-1 prediction).
        top1_confidence (float): The confidence score associated with the top-1 predicted class.
        top5_class_ids (list[int]): List of the top 5 predicted class IDs, ordered by confidence (descending).
        top5_confidences (list[float]): List of confidence scores corresponding to each class ID in `top5_class_ids`.
    """
    checkpoint = torch.load(translation_model_path, map_location=device)
    net.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    character = transform(img)
    character = character[None, :, :, :]
    output = net(character)

    probabilities = output.detach().cpu().numpy()[0]

    top1_idx = int(probabilities.argmax())
    top1_confidence = float(probabilities[top1_idx])
    
    top5_indices = probabilities.argsort()[-5:][::-1]
    top5_confidences = [float(probabilities[idx]) for idx in top5_indices]
    top5_class_ids = [int(idx) for idx in top5_indices]

    return PytorchTranslationOutput(
        top1_class_id=top1_idx,
        top1_confidence=top1_confidence,
        top5_class_ids=top5_class_ids,
        top5_confidences=top5_confidences
    )
