import os
import cv2
import numpy as np
from typing import Tuple, List
from concurrent.futures import ThreadPoolExecutor

BASE_DATA_PATH = "/mnt/pfs/loongflow/devmachine/h20-04/evolux/output/mlebench/denoising-dirty-documents/prepared/public"
OUTPUT_DATA_PATH = "output/72c59496-c374-4998-9558-a1c06c176e1b/1/executor/output"

# Task-adaptive type definitions
X = List[np.ndarray]  # List of 2D grayscale images (H, W) as float32 in [0, 1]
y = List[np.ndarray]  # List of 2D grayscale images (H, W) as float32 in [0, 1]
Ids = List[str]  # List of image filenames (without extension) for submission alignment


def _load_single_image(path: str) -> np.ndarray:
    """Loads a single image as grayscale and normalizes it."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found at: {path}")

    # Read as grayscale
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Failed to decode image at: {path}")

    # Convert to float32 and normalize to [0, 1]
    return img.astype(np.float32) / 255.0


def load_data(validation_mode: bool = False) -> Tuple[X, y, X, Ids]:
    """
    Loads and prepares the datasets for the denoising task.

    Returns:
        X_train: List of dirty images.
        y_train: List of cleaned images.
        X_test: List of dirty test images.
        test_ids: Basenames of test images.
    """
    print(f"Starting data loading. Validation mode: {validation_mode}")

    train_dir = os.path.join(BASE_DATA_PATH, "train")
    train_cleaned_dir = os.path.join(BASE_DATA_PATH, "train_cleaned")
    test_dir = os.path.join(BASE_DATA_PATH, "test")

    # Verify directory existence
    for d in [train_dir, train_cleaned_dir, test_dir]:
        if not os.path.isdir(d):
            raise RuntimeError(f"Required directory missing: {d}")

    # Get file lists
    train_files = sorted([f for f in os.listdir(train_dir) if f.endswith('.png')])
    test_files = sorted([f for f in os.listdir(test_dir) if f.endswith('.png')])

    if validation_mode:
        train_files = train_files[:10]
        test_files = test_files[:5]
        print(f"Validation mode active: subsetting to {len(train_files)} train and {len(test_files)} test images.")

    def load_train_pair(filename: str) -> Tuple[np.ndarray, np.ndarray]:
        dirty = _load_single_image(os.path.join(train_dir, filename))
        clean = _load_single_image(os.path.join(train_cleaned_dir, filename))
        return dirty, clean

    # Parallel loading for efficiency
    print(f"Loading {len(train_files)} training pairs...")
    with ThreadPoolExecutor(max_workers=min(36, len(train_files) or 1)) as executor:
        train_results = list(executor.map(load_train_pair, train_files))

    X_train = [res[0] for res in train_results]
    y_train = [res[1] for res in train_results]

    print(f"Loading {len(test_files)} test images...")
    with ThreadPoolExecutor(max_workers=min(36, len(test_files) or 1)) as executor:
        X_test = list(executor.map(lambda f: _load_single_image(os.path.join(test_dir, f)), test_files))

    # test_ids should be the filename without extension (e.g., '1' for '1.png')
    # as required by the image_row_col submission format.
    test_ids = [os.path.splitext(f)[0] for f in test_files]

    # Final validation of row alignment
    if len(X_train) != len(y_train):
        raise ValueError(f"Mismatch in train features ({len(X_train)}) and targets ({len(y_train)}).")
    if len(X_test) != len(test_ids):
        raise ValueError(f"Mismatch in test features ({len(X_test)}) and IDs ({len(test_ids)}).")

    print(f"Data loading complete. Train samples: {len(X_train)}, Test samples: {len(X_test)}")

    return X_train, y_train, X_test, test_ids