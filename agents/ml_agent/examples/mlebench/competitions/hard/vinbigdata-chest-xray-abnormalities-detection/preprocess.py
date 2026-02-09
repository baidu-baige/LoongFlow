import os
import cv2
import numpy as np
import pandas as pd
import torch
from concurrent.futures import ProcessPoolExecutor
from typing import Tuple, List, Any

# Task-adaptive type definitions
X = Any      # Will be torch.Tensor (images) or pd.DataFrame (metadata)
y = Any      # Will be List[np.ndarray] (bounding boxes)

def _preprocess_single_image(args: Tuple[str, int, int, int]) -> np.ndarray:
    """
    Worker function to apply CLAHE and Letterbox resizing to a single image.
    """
    filepath, orig_w, orig_h, target_size = args
    
    # 1. Load image in grayscale
    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"Could not read image at {filepath}")
        
    # 2. Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)
    
    # 3. Letterbox Resize (Maintain Aspect Ratio)
    # Calculate scale and padding
    scale = target_size / max(orig_h, orig_w)
    new_w, new_h = int(orig_w * scale), int(orig_h * scale)
    
    img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # Create canvas and place resized image in the center
    canvas = np.zeros((target_size, target_size), dtype=np.uint8)
    pad_x = (target_size - new_w) // 2
    pad_y = (target_size - new_h) // 2
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = img_resized
    
    return canvas

def _transform_labels(y: List[np.ndarray], X_meta: pd.DataFrame, target_size: int = 1024) -> List[np.ndarray]:
    """
    Adjusts bounding box coordinates to match the Letterbox transformation.
    """
    transformed_y = []
    for i, boxes in enumerate(y):
        if boxes.size == 0:
            transformed_y.append(boxes)
            continue
            
        orig_w = X_meta.iloc[i]['width']
        orig_h = X_meta.iloc[i]['height']
        
        scale = target_size / max(orig_h, orig_w)
        pad_x = (target_size - orig_w * scale) / 2
        pad_y = (target_size - orig_h * scale) / 2
        
        # boxes format: [x_min, y_min, x_max, y_max, class_id]
        new_boxes = boxes.copy().astype(np.float32)
        new_boxes[:, 0] = boxes[:, 0] * scale + pad_x
        new_boxes[:, 1] = boxes[:, 1] * scale + pad_y
        new_boxes[:, 2] = boxes[:, 2] * scale + pad_x
        new_boxes[:, 3] = boxes[:, 3] * scale + pad_y
        
        # Ensure boxes stay within bounds [0, target_size]
        new_boxes[:, :4] = np.clip(new_boxes[:, :4], 0, target_size)
        transformed_y.append(new_boxes)
        
    return transformed_y

def preprocess(
    X_train: pd.DataFrame,
    y_train: List[np.ndarray],
    X_val: pd.DataFrame,
    y_val: List[np.ndarray],
    X_test: pd.DataFrame
) -> Tuple[torch.Tensor, List[np.ndarray], torch.Tensor, List[np.ndarray], torch.Tensor]:
    """
    Standardizes VinBigData images using CLAHE and Letterbox resizing.
    """
    target_size = 1024
    num_workers = 36
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Starting preprocessing for {len(X_train)} train, {len(X_val)} val, and {len(X_test)} test samples.")

    def process_set(df: pd.DataFrame, desc: str) -> torch.Tensor:
        print(f"Processing {desc} images...")
        tasks = [
            (row['filepath'], row['width'], row['height'], target_size) 
            for _, row in df.iterrows()
        ]
        
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            images = list(executor.map(_preprocess_single_image, tasks))
        
        # Stack into numpy array (N, H, W)
        images_np = np.stack(images)
        
        # GPU Acceleration: Move to GPU for normalization and final tensor conversion
        # We process in chunks to avoid overwhelming GPU memory if the dataset is massive (though 440GB RAM is huge)
        tensor = torch.from_numpy(images_np).to(device)
        
        # Normalize to [0, 1] and add channel dimension (N, 1, H, W)
        tensor = tensor.unsqueeze(1).float() / 255.0
        
        # Verification: Check for NaN or Inf
        if torch.isnan(tensor).any() or torch.isinf(tensor).any():
            raise ValueError(f"NaN or Inf detected in processed {desc} tensors.")
            
        print(f"Completed {desc} image processing. Tensor shape: {tensor.shape}")
        return tensor.cpu() # Transfer back to CPU for pipeline compatibility

    # 1. Transform Images
    X_train_processed = process_set(X_train, "train")
    X_val_processed = process_set(X_val, "validation")
    X_test_processed = process_set(X_test, "test")

    # 2. Transform Bounding Boxes
    print("Adjusting bounding box coordinates...")
    y_train_processed = _transform_labels(y_train, X_train, target_size)
    y_val_processed = _transform_labels(y_val, X_val, target_size)

    # 3. Final Validation
    assert len(X_train_processed) == len(y_train_processed), "Train alignment mismatch"
    assert len(X_val_processed) == len(y_val_processed), "Val alignment mismatch"
    assert len(X_test_processed) == len(X_test), "Test coverage mismatch"

    print("Preprocessing stage completed successfully.")
    return (
        X_train_processed, 
        y_train_processed, 
        X_val_processed, 
        y_val_processed, 
        X_test_processed
    )