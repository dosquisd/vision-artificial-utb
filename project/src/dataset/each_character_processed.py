"""
Process individual braille characters with filtering.

This module provides functionality to process individual braille character images
by applying filters and saving the processed images to a specified directory.
"""

import click

import os
from pathlib import Path

import cv2
from src.filter_characters import main_filter
from src.config import settings


@click.command()
@click.argument("input_dir", type=str, required=True)
@click.argument("save_dir", type=str, required=True)
@click.option(
    "--allowed_extensions",
    type=str,
    default="",
    help="Extensiones de archivo permitidas (separadas por comas). Por defecto: png,jpg,jpeg",
)
@click.option(
    "--radius",
    type=float,
    default=1.0,
    help="Radio del filtro de máscara de desenfoque (unsharp mask). Por defecto: 1.0",
)
@click.option(
    "--amount",
    type=float,
    default=1.0,
    help="Cantidad del filtro de máscara de desenfoque (unsharp mask). Por defecto: 1.0",
)
def get_each_character_processed(
    input_dir: str, save_dir: str, allowed_extensions: str, radius: float, amount: float
) -> None:
    """
    Process braille character images by applying filters and save them to a new directory.

    This function takes a directory of braille character images, applies the main_filter
    to each image, and saves the results to the specified output directory.

    Args:
        input_dir (str): Directory containing input images to process
        save_dir (str): Directory where processed images will be saved
        allowed_extensions (str): Comma-separated list of allowed image file extensions

    Raises:
        FileNotFoundError: If the input directory does not exist
    """
    # Crear el directorio de salida si no existe
    os.makedirs(save_dir, exist_ok=True)

    # Validar la existencia de los directorios y archivos
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"La ruta de entrada: {input_dir} no existe.")

    if not len(allowed_extensions):
        allowed_extensions = settings.VALID_IMAGES_EXTENSIONS
    else:
        allowed_extensions = allowed_extensions.strip().split(",")

    input_dir: Path = Path(input_dir)
    if input_dir.is_dir():
        images_path = [list(input_dir.glob(f"*.{ext}")) for ext in allowed_extensions]
        images_path = [item for sublist in images_path for item in sublist]
    else:
        images_path = [input_dir]

    for image_path in images_path:
        basename = os.path.basename(str(image_path))
        img = main_filter(str(image_path), radius=radius, amount=amount)
        cv2.imwrite(os.path.join(save_dir, basename), img)


if __name__ == "__main__":
    get_each_character_processed()
