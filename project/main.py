"""
Main module for Braille character detection and translation.

This module provides the main entry point for the Braille character detection
and translation pipeline. It integrates character detection and translation models
to create a complete system that can:
    1. Detect Braille characters in images using YOLO models
    2. Extract individual characters from the detected regions
    3. Recognize and translate each Braille character to its text equivalent
    4. Generate an annotated output image with detected characters and translations

The system supports different model versions for both detection and translation
components, allowing for experimentation with different approaches.

As new versions of the models are released, may be added and the code may need to be
adjusted to work as expected.
"""

import cv2
from ultralytics.utils.plotting import Annotator

from src.config import settings
from src.versions import VERSIONS, utils
from src.versions.classes import (
    YOLOInput,
    InputTranslationModel,
    OutputTranslationModel,
    InputCharacterModel,
    OutputCharacterModel,
    OutputPrediction,
)
from src.filter_characters import filter
from src.dataset.train_test_processed import resize_image

from typing import Union


def main(
    *,
    image: cv2.Mat,
    character_model_version: str,
    translation_model_version: str,
    characters_kwargs: InputCharacterModel,
    translation_kwargs: InputTranslationModel,
    radius: float = 1.0,
    amount: float = 1.0,
) -> Union[OutputPrediction, None]:
    """
    Process an image to detect and translate Braille characters.

    This function applies a pipeline of Braille character detection followed by
    character translation. It supports different model versions and configurations
    for both detection and translation.

    Args:
        image (cv2.Mat): Input image containing Braille characters to process
        character_model_version (str): Version of the character detection model to use
        translation_model_version (str): Version of the translation model to use
        characters_kwargs (InputCharacterModel): Configuration parameters for the character model
        translation_kwargs (InputTranslationModel): Configuration parameters for the translation model
        radius (float, optional): Radius parameter for image filtering. Defaults to 1.0.
        amount (float, optional): Amount parameter for image filtering. Defaults to 1.0.

    Returns:
        Union[OutputPrediction, None]: Dictionary containing detection results and predicted characters,
                                      or None if invalid configuration is provided
    """
    is_character_input_yolo = utils.is_typed_dict_instance(characters_kwargs, YOLOInput)
    is_translation_input_yolo = utils.is_typed_dict_instance(
        translation_kwargs, YOLOInput
    )

    print(f"{characters_kwargs = } -- {is_character_input_yolo}")
    print(f"{translation_kwargs = } -- {is_translation_input_yolo}")

    model_path = (
        "yolo_model_path" if is_character_input_yolo else "translation_model_path"
    )
    is_model_path_yolo = utils.is_yolo_model_path(characters_kwargs[model_path])

    if is_character_input_yolo and not is_model_path_yolo:
        return None

    if not is_character_input_yolo and is_model_path_yolo:
        return None

    character_model = VERSIONS["characters"][character_model_version]
    translation_model = VERSIONS["translation"][translation_model_version]

    image = resize_image(image, settings.PROCESSED_IMAGE_SHAPE)["padded_img"]
    character_shape = (
        settings.PROCESSED_CHARACTER_SHAPE_YOLO
        if is_translation_input_yolo
        else settings.PROCESSED_CHARACTER_SHAPE
    )

    characters_kwargs["img"] = image

    out: OutputPrediction = {"boxes": [], "character_predicted": []}

    print("Prediction for characters...")
    character_output: OutputCharacterModel = character_model(**characters_kwargs)
    n = len(character_output["boxes"])
    print(n, "\n")

    annotator = Annotator(img)
    for i in range(n):
        box = character_output["boxes"][i]
        box_int = box.int().tolist()
        character_img = image[box_int[1] : box_int[3], box_int[0] : box_int[2]]
        filtered_img = filter(
            character_img, radius=radius, amount=amount, shape=character_shape
        )

        if is_translation_input_yolo:
            filtered_img = cv2.cvtColor(filtered_img, cv2.COLOR_GRAY2BGR)
        translation_kwargs["img"] = filtered_img

        print("\n\nPrediction for translation")
        translation_output: OutputTranslationModel = translation_model(
            **translation_kwargs
        )

        class_id = (
            translation_output["class_ids"][0]
            if is_translation_input_yolo
            else translation_output["class_id"]
        )

        character_predicted = settings.get_label_character(class_id)
        label_text = character_predicted
        if "confidences" in translation_output:
            conf = translation_output["confidences"][0]
            label_text = f"{label_text} - {conf:.2f}"

        # if "confidences" in character_output:
        #     conf = character_output["confidences"][i]
        #     label_text = f"{conf:.2f} - {label_text}"

        out["boxes"].append(box)
        out["character_predicted"].append(character_predicted)

        annotator.box_label(box, label_text)

    out["result"] = annotator.result()
    return out


if __name__ == "__main__":
    img = cv2.imread("./data/processed/test/images/34_1.jpg")

    # Using all yolo models
    """
    characters_kwargs = {
        "yolo_model_path": "./models/runs/detect/train2/weights/best.pt",
        "conf": 0.7,
        "iou": 0.7,
    }

    translation_kwargs = {
        "yolo_model_path": "./models/runs/translation/train1-yolo/weights/best.pt",
        "conf": 0.0,
        "iou": 0.7,
    }

    character_model_version = "v1"
    translation_model_version = "v2"
    """

    # Using yolo for characters and pytorch for translation
    characters_kwargs: InputCharacterModel = {
        "yolo_model_path": "./models/runs/detect/train2/weights/best.pt",
        "conf": 0.7,
        "iou": 0.7,
    }
    translation_kwargs: InputTranslationModel = {
        "translation_model_path": "./models/runs/translation/train4/best_model_epoch92.pth",
        "device": "cpu",
    }

    character_model_version = "v1"
    translation_model_version = "v1"

    output = main(
        image=img,
        character_model_version=character_model_version,
        translation_model_version=translation_model_version,
        characters_kwargs=characters_kwargs,
        translation_kwargs=translation_kwargs,
    )

    print("Presione cualquier tecla para salir...")
    result = output["result"]

    cv2.imshow("Result", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
