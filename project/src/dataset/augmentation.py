"""
Data augmentation module for Braille character images.

This module provides functions to augment Braille character images by applying
various transformations to simulate different real-world conditions, including:
- Changing dot and background grayscale values
- Inverting colors
- Adding noise
- Varying contrast and brightness
- Applying realistic lighting effects
- Adding blur to simulate focus issues
- Applying perspective transformations
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple
import random
from tqdm import tqdm
import click


def invert_colors(image: np.ndarray) -> np.ndarray:
    """
    Invert image colors (black to white, white to black).

    Args:
        image: Input image in grayscale or color

    Returns:
        Inverted image
    """
    return 255 - image


def change_dot_color(
    image: np.ndarray,
    dot_color: int = 255,
    background_color: int = 0,
) -> np.ndarray:
    """
    Change dot color and background color in a Braille image.

    Args:
        image: Input image
        dot_color: Grayscale value for dots (0-255)
        background_color: Grayscale value for background (0-255)

    Returns:
        Image with modified colors
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Threshold to separate dots from background
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # Create color image
    result = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

    # Set colors
    result[thresh == 255] = dot_color
    result[thresh == 0] = background_color

    return result


def add_noise(image: np.ndarray, noise_level: float = 0.05) -> np.ndarray:
    """
    Add random noise to an image.

    Args:
        image: Input image
        noise_level: Standard deviation of Gaussian noise

    Returns:
        Image with added noise
    """
    # Convert to float
    img_float = image.astype(float) / 255.0

    # Generate noise
    noise = np.random.normal(0, noise_level, img_float.shape)

    # Add noise to image
    noisy = img_float + noise

    # Clip to valid range [0, 1]
    noisy = np.clip(noisy, 0, 1)

    # Convert back to uint8
    return (noisy * 255).astype(np.uint8)


def adjust_contrast_brightness(
    image: np.ndarray, contrast: float = 1.0, brightness: int = 0
) -> np.ndarray:
    """
    Adjust contrast and brightness of an image.

    Args:
        image: Input image
        contrast: Contrast factor. Values > 1 increase contrast
        brightness: Brightness adjustment. Positive values increase brightness

    Returns:
        Adjusted image
    """
    # Apply contrast adjustment
    adjusted = cv2.convertScaleAbs(image, alpha=contrast, beta=brightness)
    return adjusted


def simulate_lighting(
    image: np.ndarray,
    direction: Tuple[float, float, float] = (0, 0, 1),
    intensity: float = 0.5,
) -> np.ndarray:
    """
    Simulate directional lighting effect on Braille dots.

    Args:
        image: Input image (grayscale or color)
        direction: 3D vector indicating light direction
        intensity: Lighting intensity factor

    Returns:
        Image with simulated lighting
    """
    # Normalize direction vector
    direction = np.array(direction)
    direction = direction / np.linalg.norm(direction)

    # Create a 2D gradient based on light direction
    y, x = np.mgrid[0 : image.shape[0], 0 : image.shape[1]]
    x_norm = x / image.shape[1] - 0.5
    y_norm = y / image.shape[0] - 0.5

    # Compute lighting effect (simplified model)
    light_effect = (
        x_norm * direction[0] + y_norm * direction[1] + direction[2]
    ) * intensity
    light_effect = np.clip(light_effect, 0, 1)

    # Convert to uint8
    light_map = (light_effect * 255).astype(np.uint8)

    # Apply lighting effect only to dots (not background)
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # Create result image
    result = image.copy()

    # Apply lighting to dots
    if len(image.shape) == 3:
        for i in range(3):
            result[:, :, i] = np.where(
                thresh == 255,
                np.minimum(result[:, :, i] + light_map, 255),
                result[:, :, i],
            )
    else:
        result = np.where(thresh == 255, np.minimum(result + light_map, 255), result)

    return result


def apply_blur(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """
    Apply Gaussian blur to simulate camera focus issues.

    Args:
        image: Input image
        kernel_size: Size of the Gaussian kernel

    Returns:
        Blurred image
    """
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)


def apply_perspective_transform(image: np.ndarray, strength: float = 0.1) -> np.ndarray:
    """
    Apply perspective transform to simulate different viewing angles.

    Args:
        image: Input image
        strength: Strength of perspective effect

    Returns:
        Transformed image
    """
    h, w = image.shape[:2]

    # Define source points
    src_points = np.array(
        [
            [0, 0],  # top-left
            [w - 1, 0],  # top-right
            [0, h - 1],  # bottom-left
            [w - 1, h - 1],  # bottom-right
        ],
        dtype=np.float32,
    )

    # Define destination points with perspective distortion
    offset_x = int(w * strength)
    offset_y = int(h * strength)

    dst_points = np.array(
        [
            [offset_x, offset_y],  # top-left
            [w - 1 - offset_x, offset_y],  # top-right
            [0, h - 1 - offset_y],  # bottom-left
            [w - 1, h - 1 - offset_y],  # bottom-right
        ],
        dtype=np.float32,
    )

    # Compute perspective transform matrix
    M = cv2.getPerspectiveTransform(src_points, dst_points)

    # Apply transformation
    return cv2.warpPerspective(
        image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
    )


def augment_image(image: np.ndarray, augmentation_type: str = "random") -> np.ndarray:
    """
    Apply augmentation to a single image based on specified type.

    Args:
        image: Input image
        augmentation_type: Type of augmentation to apply. Options include:
            - "invert": Invert colors
            - "light_dots": White dots on black background
            - "gray_dots": Gray dots on black background
            - "colored_dots": Randomly colored dots
            - "noise": Add random noise
            - "contrast": Adjust contrast and brightness
            - "lighting": Simulate directional lighting
            - "blur": Apply Gaussian blur
            - "perspective": Apply perspective transformation
            - "random": Randomly choose one of the above

    Returns:
        Augmented image
    """
    if augmentation_type == "random":
        # Choose random augmentation
        augmentation_type = random.choice(
            [
                "invert",
                "light_dots",
                "gray_dots",
                "colored_dots",
                "noise",
                "contrast",
                "lighting",
                "blur",
                "perspective",
            ]
        )

    # Apply selected augmentation
    if augmentation_type == "invert":
        return invert_colors(image)

    elif augmentation_type == "light_dots":
        # White dots on black background
        return change_dot_color(image, 255, 0)

    elif augmentation_type == "gray_dots":
        # Gray dots on black background
        gray_value = random.randint(100, 200)
        return change_dot_color(image, gray_value, 0)

    elif augmentation_type == "colored_dots":
        # Different grayscale values for dots and background
        dot_color = random.randint(100, 255)
        bg_color = random.randint(0, 50)
        return change_dot_color(image, dot_color, bg_color)

    elif augmentation_type == "noise":
        # Add random noise
        noise_level = random.uniform(0.01, 0.1)
        return add_noise(image, noise_level)

    elif augmentation_type == "contrast":
        # Adjust contrast and brightness
        contrast = random.uniform(0.7, 1.5)
        brightness = random.randint(-30, 30)
        return adjust_contrast_brightness(image, contrast, brightness)

    elif augmentation_type == "lighting":
        # Simulate directional lighting
        direction = (random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(0, 1))
        intensity = random.uniform(0.2, 0.8)
        return simulate_lighting(image, direction, intensity)

    elif augmentation_type == "blur":
        # Apply blur
        kernel_size = random.choice([3, 5])
        return apply_blur(image, kernel_size)

    elif augmentation_type == "perspective":
        # Apply perspective transform
        strength = random.uniform(0.05, 0.15)
        return apply_perspective_transform(image, strength)

    else:
        # No augmentation
        return image


def augment_dataset(
    input_dir: str,
    output_dir: str,
    augmentation_types: List[str] = ["random"],
    augmentations_per_image: int = 5,
) -> None:
    """
    Augment all images in a directory and save them to output directory.

    Args:
        input_dir: Input directory containing images
        output_dir: Output directory for augmented images
        augmentation_types: List of augmentation types to apply. For each type,
                          an augmented image will be created. If ["random"] is specified,
                          a random augmentation will be chosen for each image.
        augmentations_per_image: Number of augmented versions to create per image and
                               per augmentation type
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)

    # Get list of image files
    image_files = (
        list(input_path.glob("*.jpg"))
        + list(input_path.glob("*.jpeg"))
        + list(input_path.glob("*.png"))
    )

    print(f"Found {len(image_files)} images in {input_dir}")

    # Process each image
    for img_path in tqdm(image_files):
        # Load image
        image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)

        if image is None:
            print(f"Failed to load {img_path}")
            continue

        # Extract filename
        filename = img_path.stem
        extension = img_path.suffix

        # Generate augmented versions
        for aug_idx in range(augmentations_per_image):
            for aug_type in augmentation_types:
                # Apply augmentation
                augmented = augment_image(image, aug_type)

                # Save augmented image
                aug_name = f"{filename}_aug_{aug_idx}_{aug_type}{extension}"
                aug_path = output_path / aug_name

                cv2.imwrite(str(aug_path), augmented)

    print(
        f"Augmentation complete. Created {len(image_files) * augmentations_per_image * len(augmentation_types)} new images."
    )


@click.command()
@click.option("--input-dir", required=True, help="Directory containing source images")
@click.option("--output-dir", required=True, help="Directory to save augmented images")
@click.option(
    "--per-image", default=1, help="Number of augmentations per image", type=int
)
@click.option(
    "--augmentation-types",
    default="random",
    help="Comma-separated list of augmentation types or 'random'",
)
def main(input_dir: str, output_dir: str, per_image: int, augmentation_types: str):
    """
    Generate augmented Braille character images from a source directory.
    
    This command-line tool applies various augmentation techniques to Braille images,
    creating multiple variations to improve model training. Available augmentation
    types include: invert, light_dots, gray_dots, colored_dots, noise, contrast,
    lighting, blur, and perspective.
    """
    aug_types = augmentation_types.split(",") if augmentation_types else ["random"]
    augment_dataset(input_dir, output_dir, aug_types, per_image)


if __name__ == "__main__":
    main()
