import cv2
import numpy as np
import pandas as pd
from typing import Tuple, Any
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

# Task-adaptive type definitions
X = np.ndarray      # Model input data (Images as NumPy arrays: N x 456 x 456 x 3)
y = np.ndarray     # Learning objectives (Diagnosis labels as float32)
Ids = pd.Series      # Identifiers for output alignment (id_code)

BASE_DATA_PATH = "/root/workspace/evolux_ml/output/mlebench/aptos2019-blindness-detection/prepared/public"
OUTPUT_DATA_PATH = "output/16326c74-72ad-4b59-ad28-cc76a3d9d373/5/executor/output"

def _preprocess_single_image(path: str, size: int = 456) -> np.ndarray:
    """
    Standardizes a single retinal image:
    1. Robust Contour Crop: Isolate the retina using contour detection.
    2. Resize: Scale to 456x456 (EfficientNet-B5).
    3. Ben Graham's Enhancement: Local mean subtraction for contrast normalization.
    4. Normalization: ImageNet mean/std scaling.
    """
    # Load image
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Critical error: Image not found or corrupted at {path}")
    
    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 1. Robust Contour Crop
    # Convert to grayscale and threshold to find the retina mask
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    
    # Identify the largest circular object (the retina)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        # Find the largest contour by area
        cnt = max(contours, key=cv2.contourArea)
        # Get bounding box and crop
        x, y, w, h = cv2.boundingRect(cnt)
        img = img[y:y+h, x:x+w]
    
    # 2. Resize to 456x456
    img = cv2.resize(img, (size, size))

    # 3. Ben Graham's Enhancement (Local Mean Subtraction)
    # Formula: img = img * 4 + GaussianBlur(img, sigma=10) * -4 + 128
    # This enhances lesions and standardizes lighting across different cameras.
    # We perform this on uint8 to maintain traditional contrast enhancement properties.
    img = cv2.addWeighted(img, 4, cv2.GaussianBlur(img, (0, 0), 10), -4, 128)

    # 4. Normalization (ImageNet mean/std)
    # Convert to float32 and scale to [0, 1]
    img = img.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img - mean) / std

    return img

def _parallel_process_images(paths: pd.Series) -> np.ndarray:
    """Utilizes CPU parallelism (112 cores) to batch process images."""
    with ProcessPoolExecutor(max_workers=112) as executor:
        results = list(tqdm(
            executor.map(_preprocess_single_image, paths),
            total=len(paths),
            desc="Preprocessing retinal images"
        ))
    return np.array(results, dtype=np.float32)

def preprocess(
    X_train: pd.Series,
    y_train: pd.Series,
    X_val: pd.Series,
    y_val: pd.Series,
    X_test: pd.Series
) -> Tuple[X, y, X, y, X]:
    """
    Transforms raw retinal image paths into model-ready NumPy arrays.
    
    This function implements a robust preprocessing pipeline including contour-based
    cropping, EfficientNet-compatible resizing, Ben Graham's enhancement, and 
    ImageNet normalization. Processing is accelerated using 112 CPU workers.
    """
    
    print(f"Starting preprocessing: {len(X_train)} train, {len(X_val)} val, {len(X_test)} test.")

    # Step 1: Process images for all sets using identical deterministic logic
    # Note: Transformers are "fitted" using static ImageNet stats and fixed dimensions.
    x_train_p = _parallel_process_images(X_train)
    x_val_p = _parallel_process_images(X_val)
    x_test_p = _parallel_process_images(X_test)

    # Step 2: Ensure targets are in consistent NumPy format
    y_train_p = y_train.values.astype(np.float32)
    y_val_p = y_val.values.astype(np.float32)

    # Step 3: Validate output quality
    for name, data in [
        ("X_train", x_train_p),
        ("y_train", x_val_p),
        ("X_test", x_test_p)
    ]:
        if np.isnan(data).any() or np.isinf(data).any():
            raise ValueError(f"Preprocessed {name} contains NaN or Infinity values.")

    # Step 4: Verify structure and alignment
    assert len(x_train_p) == len(y_train_p), "Training alignment error"
    assert len(x_val_p) == len(y_val_p), "Validation alignment error"
    assert x_train_p.shape[1:] == (456, 456, 3), f"Shape mismatch: {x_train_p.shape}"
    assert x_val_p.shape[1:] == (456, 456, 3), "Validation shape mismatch"
    assert x_test_p.shape[1:] == (456, 456, 3), "Testing shape mismatch"

    return (
        x_train_p,
        y_train_p,
        x_val_p,
        y_val_p,
        x_test_p
    )