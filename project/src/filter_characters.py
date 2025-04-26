import cv2
from src.config import settings


def main_filter(image_path: str) -> cv2.Mat:
    image = cv2.imread(image_path)

    # Convertir a escala de grises
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Hacer un resize
    gray = cv2.resize(gray, settings.PROCESSED_IMAGE_SHAPE)

    # Aplizar más filtros aquí
    # ...

    return gray
