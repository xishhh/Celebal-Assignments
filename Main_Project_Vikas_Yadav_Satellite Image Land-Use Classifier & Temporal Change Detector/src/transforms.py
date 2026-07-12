"""Image augmentation and preprocessing pipelines."""

from typing import Tuple

from torchvision import transforms as T


def get_train_transforms(image_size: Tuple[int, int]) -> T.Compose:
    """Training transforms: resize + heavy augmentation + normalise.

    Args:
        image_size: (height, width) to resize inputs to.

    Returns:
        A torchvision Compose pipeline.
    """
    return T.Compose(
        [
            T.Resize(image_size),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(degrees=15),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def get_eval_transforms(image_size: Tuple[int, int]) -> T.Compose:
    """Evaluation / inference transforms: resize + normalise only.

    Args:
        image_size: (height, width) to resize inputs to.

    Returns:
        A torchvision Compose pipeline.
    """
    return T.Compose(
        [
            T.Resize(image_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
