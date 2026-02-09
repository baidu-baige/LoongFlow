import os
import cv2
import numpy as np
import torch
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from joblib import Parallel, delayed
from typing import Tuple

BASE_DATA_PATH = "/mnt/pfs/loongflow/devmachine/h20-12/evolux/output/mlebench/cassava-leaf-disease-classification/prepared/public"
OUTPUT_DATA_PATH = "output/502e395f-5bca-4b30-9a81-fc7b49a1e544/3/executor/output"

# Task-adaptive type definitions
# We utilize torch.Tensor for model-ready image data and labels.
X = torch.Tensor  # Feature matrix: (N, 3, 512, 512) image tensors
y = torch.Tensor  # Target vector: Long tensors representing class indices (0-4)

def _process_single_image(path: str, transform: A.Compose) -> torch.Tensor:
    """
    Helper function to load, transform, and convert a single image to a tensor.
    
    Args:
        path: Absolute path to the image file.
        transform: Albumentations transformation pipeline.
        
    Returns:
        torch.Tensor: Processed image tensor.
    """
    # Load image using OpenCV
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(f"Failed to load image at {path}")
    
    # Convert BGR (OpenCV default) to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Apply transformation pipeline
    augmented = transform(image=image)
    return augmented['image']

def preprocess(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame
) -> Tuple[X, y, X, y, X]:
    """
    Transforms raw image paths and labels into model-ready high-resolution tensors.
    Utilizes parallel processing (36 cores) for efficient image decoding and augmentation.
    """
    print("Initializing preprocessing pipelines...")

    # Define ImageNet normalization statistics
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]
    IMG_SIZE = 512

    # Step 1: Define Transformation Pipelines
    # Training pipeline includes heavy augmentation to combat overfitting
    train_transform = A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE, interpolation=cv2.INTER_LANCZOS4),
        A.Transpose(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=45, p=0.5),
        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.5),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2()
    ])

    # Validation/Inference pipeline: Standard resizing and normalization
    val_transform = A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE, interpolation=cv2.INTER_LANCZOS4),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2()
    ])

    # Step 2: Parallel Processing for efficient CPU utilization
    # Processing images is CPU-bound (decoding, interpolation, augmentation)
    n_jobs = 36
    
    print(f"Processing training set ({len(X_train)} samples)...")
    train_paths = X_train['image_path'].tolist()
    X_train_processed_list = Parallel(n_jobs=n_jobs)(
        delayed(_process_single_image)(path, train_transform) for path in train_paths
    )
    X_train_processed = torch.stack(X_train_processed_list)
    y_train_processed = torch.tensor(y_train.values, dtype=torch.long)

    print(f"Processing validation set ({len(X_val)} samples)...")
    val_paths = X_val['image_path'].tolist()
    X_val_processed_list = Parallel(n_jobs=n_jobs)(
        delayed(_process_single_image)(path, val_transform) for path in val_paths
    )
    X_val_processed = torch.stack(X_val_processed_list)
    y_val_processed = torch.tensor(y_val.values, dtype=torch.long)

    print(f"Processing test set ({len(X_test)} samples)...")
    test_paths = X_test['image_path'].tolist()
    X_test_processed_list = Parallel(n_jobs=n_jobs)(
        delayed(_process_single_image)(path, val_transform) for path in test_paths
    )
    X_test_processed = torch.stack(X_test_processed_list)

    # Step 3: Validate output integrity
    # Ensure no NaN or Inf values in the final tensors
    assert not torch.isnan(X_train_processed).any(), "NaN values detected in processed training features"
    assert not torch.isinf(X_train_processed).any(), "Infinity values detected in processed training features"
    assert not torch.isnan(X_val_processed).any(), "NaN values detected in processed validation features"
    assert not torch.isinf(X_val_processed).any(), "Infinity values detected in processed validation features"
    assert not torch.isnan(X_test_processed).any(), "NaN values detected in processed test features"
    assert not torch.isinf(X_test_processed).any(), "Infinity values detected in processed test features"

    # Alignment checks
    assert len(X_train_processed) == len(y_train_processed), "Training sample alignment mismatch"
    assert len(X_val_processed) == len(y_val_processed), "Validation sample alignment mismatch"
    assert len(X_test_processed) == len(X_test), "Test set completeness check failed"

    print("Preprocessing complete.")
    print(f"X_train shape: {X_train_processed.shape}, y_train shape: {y_train_processed.shape}")
    print(f"X_val shape: {X_val_processed.shape}, y_val shape: {y_val_processed.shape}")
    print(f"X_test shape: {X_test_processed.shape}")

    return (
        X_train_processed,
        y_train_processed,
        X_val_processed,
        y_val_processed,
        X_test_processed
    )