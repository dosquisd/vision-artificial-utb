"""
Braille Character Translation Model.

This module implements a CNN (Convolutional Neural Network) model for classifying
individual Braille characters. It includes the model definition, data management classes,
training routines, and auxiliary functions.

The model is designed to take grayscale images of Braille characters as input
and return the corresponding letter (a-z) that the character represents.
"""

# General use
import cv2
import numpy as np

# Path related
import os
from pathlib import Path

# CNN related
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset

from src.config import settings


beta1 = settings.BETA1
workers = settings.WORKERS
batch_size = settings.BATCH_SIZE
num_epochs = settings.NUM_EPOCHS
learning_rate = settings.LEARNING_RATE
ratio_validation = settings.RATIO_VALIDATION

transform = transforms.Compose(
    [
        transforms.ToTensor(),  # Converts (H, W, 1) to (1, H, W)
        transforms.Normalize((0.5,), (0.5,)),  # Normalizes to [-1, 1] for 1 channel
    ]
)

ngpus: int = torch.cuda.device_count()
device: torch.device = torch.device(
    "cuda:0" if (settings.USE_GPU and ngpus > 0) else "cpu"
)


class CustomDataset(Dataset):
    """
    Custom dataset for loading and processing Braille character images.

    This class implements a PyTorch Dataset that loads Braille character images
    from a directory, processes them, and labels them according to their filename.

    Attributes:
        ALLOWED_EXTENSIONS (list[str]): Allowed file extensions
        root (Path): Root directory containing the images
        transform (transforms.Compose): Transformations to apply to the images
        imgs_path (list[Path]): List of paths to all found images
    """

    ALLOWED_EXTENSIONS: list[str] = settings.VALID_IMAGES_EXTENSIONS

    def __init__(self, root: str, transform: transforms.Compose | None = None) -> None:
        """
        Initialize the dataset with a root directory and optional transformations.

        Args:
            root (str): Path to the directory containing the images
            transform (transforms.Compose, optional): Transformations to apply. Defaults to None.

        Raises:
            FileNotFoundError: If the root directory does not exist
        """
        if not os.path.exists(root):
            raise FileNotFoundError(f"Path {root} does not exist")

        self.root: Path = Path(root)
        self.transform = transform

        # Recursively search for images with allowed extensions
        images_path = [
            list(self.root.rglob(f"*{ext}")) for ext in self.ALLOWED_EXTENSIONS
        ]

        # Flatten the list of lists into a single list
        self.imgs_path: list[Path] = [
            item for sublist in images_path for item in sublist
        ]

    def map_label(self, label: str) -> int:
        """
        Maps filename to class index.

        This function extracts the label from the filename and maps it to its
        corresponding index in the lowercase alphabet.

        Args:
            label (str): The filename containing the class label

        Returns:
            int: Integer index corresponding to the character class
        """
        # Adjust this function to match your dataset's naming convention
        # and the way you want to map labels to integers.
        # Here, we assume the label is the first character of the filename.
        label = label.lower().split(".")[0]
        label = label[0]

        return settings.CHARACTERS_LOWERCASE.index(label)

    def __len__(self) -> int:
        """
        Returns the number of items in the dataset.

        Returns:
            int: Number of images in the dataset
        """
        return len(self.imgs)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get an item from the dataset by index.

        This method loads an image from disk, processes it, and returns
        it along with its corresponding label.

        Args:
            index (int): Index of the item to retrieve

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Tuple containing (image, label)
        """
        img_path = self.imgs_path[index]

        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        img = img.astype(np.float32) / 255.0  # Normalize to [0, 1]

        label = self.map_label(img_path.name)
        label = torch.tensor(label, dtype=torch.uint8)

        if self.transform:
            img = self.transform(img)
        else:
            img = torch.from_numpy(img).float()

        return img, label

    # DataLoader requires to have this property
    @property
    def imgs(self) -> list[Path]:
        """
        Property that returns the list of image paths.

        Required by PyTorch DataLoader.

        Returns:
            list[Path]: List of paths to all images in the dataset
        """
        return self.imgs_path


class BrailleCNN(nn.Module):
    """
    Convolutional Neural Network for Braille character classification.

    This model consists of three convolutional blocks followed by fully connected layers.
    Each convolutional block includes a Conv2d layer, ReLU activation, MaxPooling, and BatchNorm.
    The model is designed to classify Braille characters into lowercase letters (a-z).

    Attributes:
        model (nn.Sequential): The sequential model containing all layers
    """

    def __init__(self, num_classes=len(settings.CHARACTERS_LOWERCASE)) -> None:
        """
        Initialize the Braille CNN model.

        Args:
            num_classes (int): Number of output classes
        """
        super().__init__()
        self.model = nn.Sequential(
            # Block 1
            nn.Conv2d(
                in_channels=1, out_channels=64, kernel_size=5, stride=1, padding="same"
            ),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            # Block 2
            nn.Conv2d(
                in_channels=64, out_channels=64, kernel_size=3, stride=1, padding="same"
            ),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25),
            nn.BatchNorm2d(64),
            # Block 3
            nn.Conv2d(
                in_channels=64, out_channels=64, kernel_size=3, stride=1, padding="same"
            ),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25),
            nn.BatchNorm2d(64),
            # Fully connected layers
            nn.Flatten(),
            nn.Linear(in_features=64 * 3 * 3, out_features=576),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.BatchNorm1d(576),
            nn.Linear(in_features=576, out_features=288),
            nn.ReLU(inplace=True),
            # Output layer
            nn.Linear(in_features=288, out_features=num_classes),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor containing the Braille character image(s)

        Returns:
            torch.Tensor: Output tensor with class probabilities
        """
        return self.model(x)


net = BrailleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=learning_rate, betas=(beta1, 0.999))
