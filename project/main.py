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
import numpy as np
from ultralytics.utils.plotting import Annotator

from src.config import settings
from src.versions import utils, version_manager
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
    characters_kwargs: InputCharacterModel,
    translation_kwargs: InputTranslationModel,
    top1: bool = True,
    top5_cls_func: callable = None,
    top5_conf_func: callable = None,
    radius: float = 1.0,
    amount: float = 1.0,
) -> Union[OutputPrediction, None]:
    """
    Process an image to detect and translate Braille characters.

    This function applies a pipeline of Braille character detection followed by
    character translation. It supports different model versions and configurations
    for both detection and translation.

    Args:
        image (cv2.Mat): Input image containing Braille characters to process. Color space BGR.
        characters_kwargs (InputCharacterModel): Configuration parameters for the character model
        translation_kwargs (InputTranslationModel): Configuration parameters for the translation model
        top1 (bool, optional): Flag to indicate whether to use top-1 prediction for Translation Model. \
            Defaults to True.
        top5_cls_func (callable, optional): Function to apply top-5 results in order to get the class id. \
            This function must be defined as `top5_cls_func(cls_id: list[int], conf: list[float]) -> int`. \
            If not provided, then the top-1 prediction for class id will be used.
        top5_conf_func (callable, optional): Function to apply top-5 results in order to get the resultant confidence. \
            This function must be defined as `top5_conf_func(cls_id: list[int], conf: list[float]) -> float`. \
            If not provided, then the top-1 prediction for confidence will be used.
        radius (float, optional): Radius parameter for image filtering. Defaults to 1.0.
        amount (float, optional): Amount parameter for image filtering. Defaults to 1.0.

    Returns:
        Union[OutputPrediction, None]: Dictionary containing detection results and predicted characters,
                                      or None if invalid configuration is provided
    """
    # Valid parameters and prepare function for its correct performance
    character_model_version = characters_kwargs["version"]
    translation_model_version = translation_kwargs["version"]

    print(character_model_version, translation_model_version)

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

    if top1 or top5_cls_func is None or top5_conf_func is None:
        class_id_key = "top1_class_id"
        confidence_key = "top1_confidence"
    else:
        class_id_key = "top5_class_ids"
        confidence_key = "top5_confidences"

    if top5_cls_func is None:
        top5_cls_func = lambda cls_id, conf: cls_id[0]  # noqa: E731

    if top5_conf_func is None:
        top5_conf_func = lambda cls_id, conf: conf[0] if conf[0] is not None else None  # noqa: E731

    # Prepare output
    out: OutputPrediction = {
        "orig_img": np.array(image),
        "boxes": [],
        "character_predicted": [],
        "confidences": [],
    }

    # Attributes for character model
    character_params = {"model_type": "characters", "version": character_model_version}
    character_model = version_manager.get_model(**character_params)
    character_config = version_manager.get_config(**character_params)
    character_shape = character_config.shape
    character_color_space = character_config.color_space

    # Attributes for translation model
    translation_params = {
        "model_type": "translation",
        "version": translation_model_version,
    }
    translation_model = version_manager.get_model(**translation_params)
    translation_config = version_manager.get_config(**translation_params)
    translation_shape = translation_config.shape
    translation_color_space = translation_config.color_space

    # Preprocess image for character model
    image = resize_image(image, character_shape)["padded_img"]

    # Add more cases if needed
    if character_color_space == "RGB":
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    characters_kwargs["img"] = image

    # Make the prediction for character model
    character_output: OutputCharacterModel = character_model(**characters_kwargs)
    n = len(character_output["boxes"])
    annotator = Annotator(image)
    for i in range(n):
        box = character_output["boxes"][i]
        box_int = box.int().tolist()

        # Take character image and prepare it for its prediction
        character_img = image[box_int[1] : box_int[3], box_int[0] : box_int[2]]
        filtered_img = filter(
            character_img, radius=radius, amount=amount, shape=translation_shape
        )

        # Add more color space conversions here if needed
        if translation_color_space == "RGB":
            filtered_img = cv2.cvtColor(filtered_img, cv2.COLOR_GRAY2RGB)
        translation_kwargs["img"] = filtered_img

        # Make the prediction
        translation_output: OutputTranslationModel = translation_model(
            **translation_kwargs
        )

        # Take top1 id or top5 ids made in the previous prediction
        class_id = translation_output[class_id_key]
        if not isinstance(class_id, list):
            class_id = [class_id]

        conf = translation_output[confidence_key]
        conf = [conf] if not isinstance(conf, list) else conf
        class_id = top5_cls_func(class_id, conf)
        conf = top5_conf_func(class_id, conf)

        character_predicted = settings.get_label_character(class_id)

        label_text = character_predicted
        if conf is not None:
            label_text += f", {conf:.2f}"

        out["boxes"].append(box)
        out["character_predicted"].append(character_predicted)
        out["confidences"].append(conf)

        # Draw a box for each character and its respective text
        annotator.box_label(box, label_text)

    out["result_img"] = annotator.result()
    return out


if __name__ == "__main__":
    img = cv2.imread("./data/processed/test/images/34_1.jpg")

    # Using all yolo models
    characters_kwargs = {
        "yolo_model_path": "./models/runs/detect/train2/weights/best.pt",
        "version": "v1",
        "conf": 0.7,
        "iou": 0.7,
    }

    translation_kwargs = {
        "yolo_model_path": "./models/runs/translation/train5-yolo/weights/best.pt",
        "version": "v2",
        "conf": 0.0,
        "iou": 0.7,
    }

    extra_kwargs = {
        "top1": False,
        "top5_cls_func": lambda cls_id, conf: int(
            sum((i * conf_i) for i, conf_i in zip(cls_id, conf))
        ),
        "top5_conf_func": lambda cls_id, conf: max(conf),
    }

    """
    # Using yolo for characters and pytorch for translation
    characters_kwargs: InputCharacterModel = {
        "yolo_model_path": "./models/runs/detect/train2/weights/best.pt",
        "version": "v1",
        "conf": 0.7,
        "iou": 0.7,
    }
    translation_kwargs: InputTranslationModel = {
        "translation_model_path": "./models/runs/translation/train4/best_model_epoch92.pth",
        "version": "v1",
        "device": "cpu",
    }

    extra_kwargs = {
        "top1": False,
        "top5_cls_func": lambda cls_id, conf: int(
            sum((i * conf_i) for i, conf_i in zip(cls_id, conf))
        ),
        "top5_conf_func": lambda cls_id, conf: max(conf),
    }
    """

    output = main(
        image=img,
        characters_kwargs=characters_kwargs,
        translation_kwargs=translation_kwargs,
        **extra_kwargs,
    )

    print("Presione cualquier tecla para salir...")
    result = output["result_img"]
    result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)

    cv2.imshow("Result", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
