"""
Configuration module for the Braille Translator project.

This module defines configuration settings for image processing and model training
using Pydantic for validation and environment variable management.
"""

from string import ascii_lowercase

from pydantic import computed_field, BeforeValidator
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Annotated

from dotenv import load_dotenv

load_dotenv()


def split_list(value: str) -> list[str]:
    """
    Splits a string into a list of strings based on commas.

    Args:
        value (str): The input value to be split.

    Returns:
        list[str] | str: A list of strings if the input is a comma-separated string,
                         otherwise returns the input value as is.
    """
    return [item.strip() for item in value.split(",")]


class Settings(BaseSettings):
    """
    Application settings for image processing dimensions and configurations.

    This class uses Pydantic to manage configuration with environment variable support,
    providing dimensions for processed images and character regions.
    """

    model_config = SettingsConfigDict(
        env_file=".env", env_ignore_empty=True, extra="ignore"
    )

    # It's recommended to use ABSOLUTE PATH in the .env file
    SAVE_PATH: str

    VALID_IMAGES_EXTENSIONS: Annotated[list[str] | str, BeforeValidator(split_list)] = [
        ".jpg",
        ".jpeg",
        ".png",
    ]

    IMAGES_DIR: str = "images"
    ANNOTATIONS_DIR: str = "labels"

    PROCESSED_IMAGE_WIDTH: int
    PROCESSED_IMAGE_HEIGHT: int

    @computed_field
    @property
    def PROCESSED_IMAGE_SHAPE(self) -> tuple[int, int]:
        """
        Returns the dimensions of processed images as a tuple (width, height).

        Returns:
            tuple[int, int]: Width and height of processed images
        """
        return self.PROCESSED_IMAGE_WIDTH, self.PROCESSED_IMAGE_HEIGHT

    PROCESSED_CHARACTER_HEIGHT: int
    PROCESSED_CHARACTER_WIDTH: int

    @computed_field
    @property
    def PROCESSED_CHARACTER_SHAPE(self) -> tuple[int, int]:
        """
        Returns the dimensions of processed character regions as a tuple (width, height).

        Returns:
            tuple[int, int]: Width and height of processed character regions
        """
        return self.PROCESSED_CHARACTER_HEIGHT, self.PROCESSED_CHARACTER_WIDTH

    PROCESSED_CHARACTER_HEIGHT_YOLO: int
    PROCESSED_CHARACTER_WIDTH_YOLO: int

    @computed_field
    @property
    def PROCESSED_CHARACTER_SHAPE_YOLO(self) -> tuple[int, int]:
        """
        Returns the dimensions of processed character regions for YOLO as a tuple (width, height).

        Returns:
            tuple[int, int]: Width and height of processed character regions for YOLO
        """
        return self.PROCESSED_CHARACTER_HEIGHT_YOLO, self.PROCESSED_CHARACTER_WIDTH_YOLO

    BETA1: float
    WORKERS: int
    BATCH_SIZE: int
    NUM_EPOCHS: int
    MINI_BATCHES: int
    LEARNING_RATE: float
    RATIO_VALIDATION: float
    CHARACTERS_LOWERCASE: str = ascii_lowercase

    USE_GPU: bool = False

    # Change this method based on the dataset
    @classmethod
    def get_class_name(cls, filename: str) -> str:
        """
        Returns the class name based on the filename.

        Args:
            filename (str): The name of the file.

        Returns:
            str: The class name derived from the filename.
        """
        return filename.split(".")[0][0]


settings = Settings()


if __name__ == "__main__":
    print(settings.model_dump())
