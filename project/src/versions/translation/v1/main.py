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
        PytorchTranslationOutput: Object containing translation result with the following attribute:
            - class_id: Predicted class ID for the input character.
    """
    checkpoint = torch.load(translation_model_path, map_location=device)
    net.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    character = transform(img)
    character = character[None, :, :, :]
    output = net(character)
    _, label_prediction = torch.max(output, 1)

    return PytorchTranslationOutput(class_id=label_prediction)
