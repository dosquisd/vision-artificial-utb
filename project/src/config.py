from pydantic import computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict

from os import getenv
from dotenv import load_dotenv

load_dotenv()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_ignore_empty=True, extra="ignore"
    )

    PROCESSED_IMAGE_WIDTH: int = 640
    PROCESSED_IMAGE_HEIGHT: int = 640

    @computed_field
    @property
    def PROCESSED_IMAGE_SHAPE(self) -> tuple[int, int]:
        return self.PROCESSED_IMAGE_WIDTH, self.PROCESSED_IMAGE_HEIGHT


settings = Settings()