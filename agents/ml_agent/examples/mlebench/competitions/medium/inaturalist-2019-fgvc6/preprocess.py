import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms.autoaugment import AutoAugmentPolicy
from PIL import Image
from typing import Tuple, Any

# Concrete type definitions for this task
X = Any  # Will be an instance of INatDataset (torch.utils.data.Dataset)
y = Any  # Will be a pd.Series containing category_id labels


class INatDataset(Dataset):
    """
    A custom Dataset for loading iNaturalist images and applying transformations.
    """

    def __init__(self, X_df: pd.DataFrame, y_series: pd.Series = None,
                 transform: Any = None, mixup_alpha: float = 0.0, cutmix_alpha: float = 0.0
                 ):
        """
        Args:
            X_df: DataFrame containing at least a 'path' column.
            y_series: Series containing labels aligned with inputs_df.
            transform: Torchvision transforms to apply to images.
            mixup_alpha: Parameter for Mixup augmentation (to be used in training loop).
            cutmix_alpha: Parameter for CutMix augmentation (to be used in training loop).
        """
        self.paths = X_df['path'].values
        self.labels = y_series.values if y_series is not None else None
        self.transform = transform
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha

        # Store metadata (taxonomy) if available
        self.metadata = X_df.drop(columns=['path']) if 'path' in X_df.columns else X_df

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Any:
        img_path = self.paths[idx]

        # Load image; let errors propagate if file is missing or corrupt
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        if self.labels is not None:
            # Return image and label as a long tensor
            return image, torch.tensor(self.labels[idx], dtype=torch.long)

        return image


def preprocess(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame
) -> Tuple[X, y, X, y, X]:
    """
    Transforms raw image paths and labels into model-ready Dataset objects.
    """

    # ImageNet normalization statistics
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    # Define training transformations according to instructions
    train_transform = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.RandomResizedCrop(384, scale=(0.08, 1.0), ratio=(0.75, 1.33)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.AutoAugment(AutoAugmentPolicy.IMAGENET),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    # Define validation and test transformations
    val_transform = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.CenterCrop(384),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    # Create transformed datasets
    # Mixup and CutMix parameters are attached to the training dataset metadata
    X_train_p = INatDataset(
        X_df=X_train,
        y_series=y_train,
        transform=train_transform,
        mixup_alpha=0.8,
        cutmix_alpha=1.0
    )

    X_val_p = INatDataset(
        X_df=X_val,
        y_series=y_val,
        transform=val_transform
    )

    X_test_p = INatDataset(
        X_df=X_test,
        y_series=None,
        transform=val_transform
    )

    # y are returned as is (pd.Series), as they are already aligned and model-ready
    # The Dataset objects also contain the labels for convenience in the PyTorch training loop.
    return (
        X_train_p,
        y_train,
        X_val_p,
        y_val,
        X_test_p
    )