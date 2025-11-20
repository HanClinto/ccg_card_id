"""
Image processing utilities
"""

from typing import Tuple, Union

import numpy as np
import torch
from PIL import Image
from torchvision import transforms


def load_image(image_path: str, size: Tuple[int, int] = None) -> Image.Image:
    """
    Load an image from disk

    Args:
        image_path: Path to the image file
        size: Optional target size (width, height)

    Returns:
        PIL Image object
    """
    img = Image.open(image_path).convert("RGB")
    if size:
        img = img.resize(size, Image.LANCZOS)
    return img


def preprocess_image(
    image: Union[Image.Image, str],
    size: int = 224,
    normalize: bool = True,
) -> torch.Tensor:
    """
    Preprocess image for model input

    Args:
        image: PIL Image or path to image
        size: Target size for resizing
        normalize: Whether to apply ImageNet normalization

    Returns:
        Preprocessed image tensor
    """
    if isinstance(image, str):
        image = load_image(image)
    
    transform_list = [
        transforms.Resize((size, size)),
        transforms.ToTensor(),
    ]
    
    if normalize:
        transform_list.append(
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        )
    
    transform = transforms.Compose(transform_list)
    return transform(image)


def preprocess_batch(
    images: list,
    size: int = 224,
    normalize: bool = True,
) -> torch.Tensor:
    """
    Preprocess a batch of images

    Args:
        images: List of PIL Images or paths
        size: Target size for resizing
        normalize: Whether to apply ImageNet normalization

    Returns:
        Batch tensor of preprocessed images
    """
    preprocessed = []
    for img in images:
        preprocessed.append(preprocess_image(img, size, normalize))
    
    return torch.stack(preprocessed)


def compute_image_hash(image: Image.Image, hash_size: int = 8) -> str:
    """
    Compute perceptual hash of an image

    Args:
        image: PIL Image
        hash_size: Size of the hash

    Returns:
        Hex string representation of the hash
    """
    import imagehash
    return str(imagehash.phash(image, hash_size=hash_size))


def resize_image_keeping_aspect(
    image: Image.Image,
    target_size: int = 224,
    pad_color: Tuple[int, int, int] = (0, 0, 0),
) -> Image.Image:
    """
    Resize image while keeping aspect ratio, padding to square

    Args:
        image: PIL Image
        target_size: Target size for the square output
        pad_color: RGB color for padding

    Returns:
        Resized and padded image
    """
    # Calculate scaling factor
    width, height = image.size
    scale = target_size / max(width, height)
    
    new_width = int(width * scale)
    new_height = int(height * scale)
    
    # Resize image
    resized = image.resize((new_width, new_height), Image.LANCZOS)
    
    # Create padded image
    padded = Image.new("RGB", (target_size, target_size), pad_color)
    
    # Paste resized image in center
    x_offset = (target_size - new_width) // 2
    y_offset = (target_size - new_height) // 2
    padded.paste(resized, (x_offset, y_offset))
    
    return padded
