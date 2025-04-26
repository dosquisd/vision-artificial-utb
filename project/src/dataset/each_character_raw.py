"""
Extract individual braille characters from labeled images.

This module provides functionality to extract individual braille character images
from source images and their corresponding label files with bounding box coordinates.
"""

import click

import os
from pathlib import Path

import numpy as np
from PIL import Image

ANNOTATIONS_DIR: str = "labels"


@click.command()
@click.argument("input_dir", type=str, required=True)
@click.argument("save_dir", type=str, required=True)
@click.option(
    "--allowed_extensions",
    type=str,
    default="png,jpg,jpeg",
    help="Extensiones de archivo permitidas (separadas por comas). Por defecto: png,jpg,jpeg",
)
def get_each_character_raw(
    input_dir: str, save_dir: str, allowed_extensions: str
) -> None:
    """
    Extract individual braille characters from images based on bounding box labels.

    This function processes images and their corresponding label files containing
    bounding box coordinates in YOLO format, extracts each character using the
    coordinates, and saves them as individual image files.

    Args:
        input_dir (str): Directory containing input images and label files
        save_dir (str): Directory where extracted character images will be saved
        allowed_extensions (str): Comma-separated list of allowed image file extensions

    Raises:
        FileNotFoundError: If the input directory does not exist
    """
    # Crear el directorio de salida si no existe
    os.makedirs(save_dir, exist_ok=True)

    # Validar la existencia de los directorios y archivos
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"La ruta de entrada: {input_dir} no existe.")

    allowed_extensions = allowed_extensions.strip().split(",")

    input_dir: Path = Path(input_dir)

    if input_dir.is_dir():
        images_path = [list(input_dir.glob(f"*.{ext}")) for ext in allowed_extensions]
        images_path = [item for sublist in images_path for item in sublist]
    else:
        images_path = [input_dir]

    for image_path in images_path:
        basename = os.path.basename(str(image_path))
        filename, ext = os.path.splitext(basename)
        parent = image_path.parents[1]
        label_path = parent / f"{ANNOTATIONS_DIR}/{filename}.txt"

        with open(label_path, "r") as labels_txt:
            labels = list(
                map(lambda label: label.strip().split(), labels_txt.readlines())
            )
            boxes = [label[1:] for label in labels]

        img = Image.open(str(image_path))
        img = np.array(img)
        x, y = img.shape[:2]
        for i, box in enumerate(boxes):
            center_x = float(box[0]) * y
            center_y = float(box[1]) * x
            w = float(box[2]) * y
            h = float(box[3]) * x

            x1 = int(center_x - w / 2)
            y1 = int(center_y - h / 2)
            x2 = int(center_x + w / 2)
            y2 = int(center_y + h / 2)

            if x1 > x2:
                x1, x2 = x2, x1
            if y1 > y2:
                y1, y2 = y2, y1

            character = img[y1:y2, x1:x2]
            output_path = os.path.join(save_dir, f"{filename}_{i}{ext}")
            try:
                Image.fromarray(character).save(output_path)
            except Exception as e:
                print(f"Error al guardar la imagen {output_path}: {e}", end="  ---  ")
                print(f"{x1=}, {y1=}, {x2=}, {y2=}")


if __name__ == "__main__":
    get_each_character_raw()
