import click

import os
from pathlib import Path

import cv2
from src.filter_characters import main_filter


@click.command()
@click.argument('input_dir', type=str, required=True)
@click.argument('save_dir', type=str, required=True)
@click.option('--allowed_extensions', type=str, default='png,jpg,jpeg', help='Extensiones de archivo permitidas (separadas por comas). Por defecto: png,jpg,jpeg')
def get_each_character_processed(
    input_dir: str,
    save_dir: str,
    allowed_extensions: str
) -> None:
    # Crear el directorio de salida si no existe
    os.makedirs(save_dir, exist_ok=True)
    
    # Validar la existencia de los directorios y archivos
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"La ruta de entrada: {input_dir} no existe.")

    allowed_extensions = allowed_extensions.strip().split(',')

    input_dir: Path = Path(input_dir)
    if input_dir.is_dir():
        images_path = [list(input_dir.glob(f"*.{ext}")) for ext in allowed_extensions]
        images_path = [item for sublist in images_path for item in sublist]
    else:
        images_path = [input_dir]

    for image_path in images_path:
        basename = os.path.basename(str(image_path))
        img = main_filter(str(image_path))
        cv2.imwrite(os.path.join(save_dir, basename), img)


if __name__ == "__main__":
    get_each_character_processed()
