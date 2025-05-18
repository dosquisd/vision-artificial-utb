"""
Resize and normalize training and testing images for braille detection model.

This module provides functionality to resize images and adapt their annotations
to a standardized size while maintaining aspect ratio with padding.
"""

import cv2
import shutil
import os
import xml.etree.ElementTree as ET
import click
from src.config import settings

image_dir: str = settings.IMAGES_DIR
annotations_dir: str = settings.ANNOTATIONS_DIR
valid_extensions: tuple = settings.VALID_IMAGES_EXTENSIONS


def resize_images_and_annotations(
    input_dir: str, output_dir: str, target_size: tuple = None, class_name_fn: callable = None
) -> None:
    """
    Resize images and adjust their XML annotations to a standard size.

    This function processes images and their corresponding XML annotations,
    resizing them to a standard size while preserving aspect ratio.
    Padding is added where necessary, and bounding box coordinates are
    adjusted accordingly.

    Args:
        input_dir (str): Directory containing original images and annotations
        output_dir (str): Directory where resized images and adjusted annotations will be saved
        target_size (tuple): Target size for the images (width, height). If None, uses default from settings.
        class_name_fn (callable): Function to determine the class name for each object in the XML annotations.
    """
    if target_size is None:
        target_size = settings.PROCESSED_IMAGE_SHAPE

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, image_dir), exist_ok=True)
    os.makedirs(os.path.join(output_dir, annotations_dir), exist_ok=True)

    for filename in os.listdir(os.path.join(input_dir, image_dir)):
        if not filename.lower().endswith(tuple(valid_extensions)):
            continue

        # Procesar imagen
        img_path = os.path.join(input_dir, image_dir, filename)
        img = cv2.imread(img_path)
        original_h, original_w = img.shape[:2]

        # Redimensionar imagen manteniendo relación de aspecto
        scale = min(target_size[0] / original_w, target_size[1] / original_h)
        new_w = int(original_w * scale)
        new_h = int(original_h * scale)

        resized_img = cv2.resize(img, (new_h, new_w))

        # Añadir padding si es necesario
        delta_w = target_size[0] - new_w
        delta_h = target_size[1] - new_h
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)

        padded_img = cv2.copyMakeBorder(
            resized_img,
            top,
            bottom,
            left,
            right,
            cv2.BORDER_CONSTANT,
            value=(0, 0, 0),
        )

        # Guardar imagen redimensionada
        cv2.imwrite(os.path.join(output_dir, image_dir, filename), padded_img)

        # Procesar anotaciones XML correspondientes
        xml_filename = os.path.splitext(filename)[0] + ".xml"
        xml_path = os.path.join(input_dir, annotations_dir, xml_filename)

        if not os.path.exists(xml_path):
            shutil.copyfile(
                os.path.join(input_dir, annotations_dir, xml_filename),
                os.path.join(output_dir, annotations_dir, xml_filename),
            )

        tree = ET.parse(xml_path)
        root = tree.getroot()

        # Actualizar tamaño de imagen en el XML
        size = root.find("size")
        size.find("width").text = str(target_size[0])
        size.find("height").text = str(target_size[1])

        # Actualizar coordenadas de cada objeto
        for obj in root.iter("object"):
            obj.find("name").text = class_name_fn(filename) if class_name_fn else "0"

            bbox = obj.find("bndbox")
            xmin = int(bbox.find("xmin").text)
            ymin = int(bbox.find("ymin").text)
            xmax = int(bbox.find("xmax").text)
            ymax = int(bbox.find("ymax").text)

            # Ajustar coordenadas al redimensionamiento
            xmin = int(xmin * scale) + left
            ymin = int(ymin * scale) + top
            xmax = int(xmax * scale) + left
            ymax = int(ymax * scale) + top

            # Asegurar que no salen de los límites
            xmin = max(0, min(xmin, target_size[0] - 1))
            xmax = max(0, min(xmax, target_size[0] - 1))
            ymin = max(0, min(ymin, target_size[1] - 1))
            ymax = max(0, min(ymax, target_size[1] - 1))

            bbox.find("xmin").text = str(xmin)
            bbox.find("ymin").text = str(ymin)
            bbox.find("xmax").text = str(xmax)
            bbox.find("ymax").text = str(ymax)

        # Guardar XML modificado
        tree.write(os.path.join(output_dir, annotations_dir, xml_filename))


@click.command()
@click.option(
    "--input_dir",
    type=str,
    required=True,
    help="Directorio de entrada con imágenes y anotaciones",
)
@click.option(
    "--output_dir",
    type=str,
    required=True,
    help="Directorio de salida para imágenes y anotaciones redimensionadas",
)
def main(input_dir: str, output_dir: str, target_size: tuple = None) -> None:
    """
    Main function to execute the resizing and annotation adjustment.

    Args:
        input_dir (str): Directory containing original images and annotations
        output_dir (str): Directory where resized images and adjusted annotations will be saved
        target_size (tuple): Target size for the images (width, height). If None, uses default from settings.
    """
    resize_images_and_annotations(input_dir, output_dir, target_size)


if __name__ == "__main__":
    main()
