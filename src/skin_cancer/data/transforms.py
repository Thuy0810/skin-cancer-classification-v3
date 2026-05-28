from __future__ import annotations

import albumentations as A
from albumentations.pytorch import ToTensorV2

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_train_transforms(
    image_size: int,
    horizontal_flip_p: float = 0.5,
    vertical_flip_p: float = 0.2,
    rotate_limit: int = 20,
    brightness_contrast_p: float = 0.4,
    shift_scale_rotate_p: float = 0.4,
) -> A.Compose:
    """Augmentations used only for training."""
    return A.Compose(
        [
            A.Resize(height=image_size, width=image_size),
            A.HorizontalFlip(p=horizontal_flip_p),
            A.VerticalFlip(p=vertical_flip_p),
            A.Rotate(limit=rotate_limit, border_mode=0, p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.05,
                scale_limit=0.10,
                rotate_limit=rotate_limit,
                border_mode=0,
                p=shift_scale_rotate_p,
            ),
            A.RandomBrightnessContrast(p=brightness_contrast_p),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )


def get_valid_transforms(image_size: int) -> A.Compose:
    """Deterministic transforms for validation, test and inference."""
    return A.Compose(
        [
            A.Resize(height=image_size, width=image_size),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )
