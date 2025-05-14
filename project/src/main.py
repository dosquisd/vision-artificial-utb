# TODO: refactor this file and src directory, in order to make centralized variables

import cv2
from PIL import Image
import numpy as np

import torch
from ultralytics import YOLO
from src.filter_characters import filter
from src.modeling import translation_model
from ultralytics.utils.plotting import Annotator

from copy import deepcopy
from string import ascii_lowercase


def get_label_character(index: int, characters_lowercase: str = ascii_lowercase) -> str:
    """
    Get the label of a character based on its index.

    Args:
        index (int): Index of the character
        characters_lowercase (str): String containing lowercase characters

    Returns:
        str: The corresponding character label
    """
    if index < 0 or index >= len(characters_lowercase):
        raise ValueError("Index out of range")
    return characters_lowercase[index]


def main(
    image_path: str,
    yolo_model_path: str = "../models/runs/detect/train/weights/best.pt",
    translation_model_path: str = "../models/runs/translation/train4/best_model_epoch92.pth",
    device: str = "cpu",
    radius: float = 1.0,
    amount: float = 1.0,
) -> None:
    # Load YOLOv11 model
    yolo_model = YOLO(
        model=yolo_model_path,
        task="detect",
    )

    # Load translation model
    net = deepcopy(translation_model.net)
    optimizer = deepcopy(translation_model.optimizer)
    transform = translation_model.transform
    net.eval()

    checkpoint = torch.load(translation_model_path, map_location=device)
    net.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # Load image
    img = Image.open(image_path)
    img_arr = np.array(img)
    predictions = yolo_model.predict(source=img, conf=0.7)[0]
    n = len(predictions.boxes)

    annotator = Annotator(img)
    for i in range(n):
        box = predictions.boxes.xyxy[i]
        box_int = box.int().tolist()
        character = img_arr[box_int[1] : box_int[3], box_int[0] : box_int[2]]
        character = cv2.cvtColor(character, cv2.COLOR_RGB2BGR)
        character = filter(character, radius=radius, amount=amount)

        # Translate character
        character = transform(character)
        character = character[None, :, :, :]
        output = net(character)
        _, label_prediction = torch.max(output, 1)
        character_prediction = get_label_character(label_prediction.item())

        # Draw bounding box and label on the image
        annotator.box_label(box, character_prediction)
    
    result = annotator.result()
    result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)

    print("Presione cualquier tecla para salir...")

    cv2.imshow("Result", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main(
        image_path="./data/processed/test/images/2_0.jpg",
        yolo_model_path="./models/runs/detect/train/weights/best.pt",
        translation_model_path="./models/runs/translation/train4/best_model_epoch92.pth",
        device="cpu",
        radius=1.0,
        amount=1.0,
    )
