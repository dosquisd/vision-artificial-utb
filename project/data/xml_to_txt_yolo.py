from pathlib import Path
import xml.etree.ElementTree as ET
from typing import TypedDict
import click
import os.path


class Size(TypedDict):
    """Represents the size of an image."""

    width: int
    height: int


class Box(TypedDict):
    """Represents a bounding box in the XML format."""

    xmin: int
    ymin: int
    xmax: int
    ymax: int
    class_name: str


class YoloBox(TypedDict):
    """Represents a bounding box in the YOLO format."""

    x: float
    y: float
    width: float
    height: float
    class_id: int


def convert(size: Size, box: Box, classes_list: list[str]) -> YoloBox:
    """
    Converts a bounding box from XML format to YOLO format.

    Args:
        size (Size): The size of the image (width and height).
        box (Box): The bounding box in XML format.
        classes_list (list[str]): List of class names.

    Returns:
        YoloBox: The bounding box in YOLO format.
    """
    inverted_width = 1.0 / size["width"]
    inverted_height = 1.0 / size["height"]
    x = (box["xmin"] + box["xmax"]) / 2
    y = (box["ymin"] + box["ymax"]) / 2
    width = box["xmax"] - box["xmin"]
    height = box["ymax"] - box["ymin"]

    x *= inverted_width
    width *= inverted_width
    y *= inverted_height
    height *= inverted_height

    try:
        class_id = classes_list.index(box["class_name"])
    except ValueError:
        class_id = -1

    return {
        "x": round(x, 6),
        "y": round(y, 6),
        "width": round(width, 6),
        "height": round(height, 6),
        "class_id": round(class_id, 6),
    }


def create_txt(filename: str, yolo_boxes: list[YoloBox]) -> None:
    """
    Creates a YOLO format text file from a list of YOLO bounding boxes.

    Args:
        filename (str): The name of the output text file.
        yolo_boxes (list[YoloBox]): List of bounding boxes in YOLO format.

    Returns:
        None
    """
    if os.path.exists(filename):
        print(f"File {filename} already exists, skipping creation.")
        return None

    with open(filename, "x") as f:
        for box in yolo_boxes:
            f.write(
                f"{box['class_id']} {box['x']:.6f} {box['y']:.6f} {box['width']:.6f} {box['height']:.6f}\n"
            )


def main(dir_path: str, classes: str) -> None:
    """
    Main function to convert XML annotations to YOLO format.

    Args:
        dir_path (str): Path to the directory containing XML files.
        classes (str): Comma-separated list of class names.

    Returns:
        None
    """
    classes_list = classes.split(",")

    path = Path(dir_path)
    xml_files = list(path.rglob("*.xml"))
    for xml_file in xml_files:
        xml_file = str(xml_file)
        try:
            # Using ElementTree instead of minidom for better error handling
            tree = ET.parse(xml_file)
            root = tree.getroot()

            # Get image size
            size_elem = root.find("size")
            width = int(size_elem.find("width").text)
            height = int(size_elem.find("height").text)

            # Process all objects (bounding boxes)
            tmp_yolobox = []
            for obj in root.findall("object"):
                bndbox = obj.find("bndbox")
                xmin = int(bndbox.find("xmin").text)
                ymin = int(bndbox.find("ymin").text)
                xmax = int(bndbox.find("xmax").text)
                ymax = int(bndbox.find("ymax").text)
                class_name = obj.find("name").text

                size_dict = Size(width=width, height=height)
                box_dict = Box(
                    xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax, class_name=class_name
                )
                yolo_box = convert(size_dict, box_dict, classes_list)
                tmp_yolobox.append(yolo_box)

            txt_file = xml_file.replace(".xml", ".txt")
            create_txt(txt_file, tmp_yolobox)
        except ET.ParseError as e:
            print(f"Error parsing XML file {xml_file}: {e}")
            continue
        except Exception as e:
            print(f"Error processing file {xml_file}: {e}")
            continue


@click.command()
@click.argument(
    "dir_path",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    default="./project/data/processed",
)
@click.option(
    "-c",
    "--classes",
    type=str,
    help="Classes to be used in the conversion. Separated by commas.",
    default="Character",
)
def main_cli(dir_path: str, classes: str) -> None:
    """
    Command-line interface for the XML to YOLO conversion script.

    Args:
        dir_path (str): Path to the directory containing XML files.
        classes (str): Comma-separated list of class names.
    """
    main(dir_path, classes)


if __name__ == "__main__":
    """
    Entry point for the script. Parses command-line arguments and starts the conversion process.
    """
    main_cli()
