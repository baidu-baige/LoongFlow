import pandas as pd
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import os
import pickle
from typing import Tuple

# Task-adaptive type definitions
X = pd.DataFrame  # Features: contains image_path
y = pd.Series     # Target: encoded hotel_id

# Path Configuration
OUTPUT_DATA_PATH = "output/6c275358-248b-46e3-a3f8-feb17fef7b7f/3/executor/output"

def preprocess(
    X_train: X,
    y_train: y,
    X_val: X,
    y_val: y,
    X_test: X
) -> Tuple[X, y, X, y, X]:
    """
    Transforms raw metadata into model-ready format for the Hotel Recognition task.
    
    This function:
    1. Defines a high-resolution (384x384) Albumentations pipeline to preserve fine-grained details.
    2. Implements aspect-ratio aware resizing using LongestMaxSize and PadIfNeeded.
    3. Persists the augmentation pipelines to disk for the downstream training stage.
    4. Ensures feature consistency and row alignment across all data splits.
    """
    print("Stage 3: Preprocessing metadata and defining 384x384 augmentation pipelines...")

    # 1. Define Albumentations Pipelines
    # Technical Specification: Resize 384x384 via LongestMaxSize + PadIfNeeded (value=0)
    # Technical Specification: Augs include Flip, ShiftScaleRotate, BrightnessContrast, HueSaturation, CoarseDropout
    # Technical Specification: Normalization uses ImageNet stats; Output is torch tensors.
    
    common_resize = [
        A.LongestMaxSize(max_size=384),
        A.PadIfNeeded(min_height=384, min_width=384, border_mode=cv2.BORDER_CONSTANT, value=0),
    ]
    
    normalization = A.Normalize(
        mean=(0.485, 0.456, 0.406), 
        std=(0.229, 0.224, 0.225)
    )

    train_transform = A.Compose([
        *common_resize,
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
        A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.5),
        normalization,
        ToTensorV2(),
    ])

    val_test_transform = A.Compose([
        *common_resize,
        normalization,
        ToTensorV2(),
    ])

    # 2. Persist transformation pipelines
    # We save these to the executor output directory for the 'train_and_predict' stage.
    os.makedirs(OUTPUT_DATA_PATH, exist_ok=True)
    
    train_pkl_path = os.path.join(OUTPUT_DATA_PATH, "train_transform.pkl")
    val_pkl_path = os.path.join(OUTPUT_DATA_PATH, "val_transform.pkl")
    
    with open(train_pkl_path, "wb") as f:
        pickle.dump(train_transform, f)
    with open(val_pkl_path, "wb") as f:
        pickle.dump(val_test_transform, f)
    
    print(f"Augmentation pipelines saved to {OUTPUT_DATA_PATH}")

    # 3. Finalize Data Structures
    # Requirement: Column consistency - all transformed sets must have identical structure.
    # The primary feature for the vision model is 'image_path'.
    cols_to_keep = ['image_path']
    
    X_train_processed = X_train[cols_to_keep].reset_index(drop=True)
    y_train_processed = y_train.reset_index(drop=True)
    
    X_val_processed = X_val[cols_to_keep].reset_index(drop=True)
    y_val_processed = y_val.reset_index(drop=True)
    
    X_test_processed = X_test[cols_to_keep].reset_index(drop=True)

    # 4. Validation and alignment checks
    # Row alignment check
    assert len(X_train_processed) == len(y_train_processed), "Alignment Error: Train features/targets mismatch."
    assert len(X_val_processed) == len(y_val_processed), "Alignment Error: Val features/targets mismatch."
    
    # Test completeness check
    assert len(X_test_processed) == len(X_test), "Completeness Error: Test samples were dropped."
    
    # Null check
    if X_train_processed.isnull().any().any() or X_test_processed.isnull().any().any():
        raise ValueError("Preprocessing resulted in unexpected NaN values.")

    print(f"Preprocessing complete. Image resolution set to 384x384.")
    print(f"Sample counts: Train={len(X_train_processed)}, Val={len(X_val_processed)}, Test={len(X_test_processed)}")

    return X_train_processed, y_train_processed, X_val_processed, y_val_processed, X_test_processed