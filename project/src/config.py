"""
Configuration module for the Braille Translator project.

This module defines configuration settings for image processing and model training
using Pydantic for validation and environment variable management.
"""

from pydantic import computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict

from dotenv import load_dotenv

load_dotenv()


class Settings(BaseSettings):
    """
    Application settings for image processing dimensions and configurations.

    This class uses Pydantic to manage configuration with environment variable support,
    providing dimensions for processed images and character regions.
    """

    model_config = SettingsConfigDict(
        env_file=".env", env_ignore_empty=True, extra="ignore"
    )

    PROCESSED_IMAGE_WIDTH: int = 640
    PROCESSED_IMAGE_HEIGHT: int = 640

    @computed_field
    @property
    def PROCESSED_IMAGE_SHAPE(self) -> tuple[int, int]:
        """
        Returns the dimensions of processed images as a tuple (width, height).

        Returns:
            tuple[int, int]: Width and height of processed images
        """
        return self.PROCESSED_IMAGE_WIDTH, self.PROCESSED_IMAGE_HEIGHT

    PROCESSED_CHARACTER_HEIGHT: int = 125
    PROCESSED_CHARACTER_WIDTH: int = 80

    @computed_field
    @property
    def PROCESSED_CHARACTER_SHAPE(self) -> tuple[int, int]:
        """
        Returns the dimensions of processed character regions as a tuple (width, height).

        Returns:
            tuple[int, int]: Width and height of processed character regions
        """
        return self.PROCESSED_CHARACTER_HEIGHT, self.PROCESSED_CHARACTER_WIDTH


settings = Settings()
