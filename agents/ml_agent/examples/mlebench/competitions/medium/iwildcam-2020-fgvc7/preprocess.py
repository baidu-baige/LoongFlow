import torch
from torchvision import transforms
from PIL import Image, ImageFile
import pandas as pd
import numpy as np
import os
import logging
from typing import Tuple, Any

# Ensure truncated images can be loaded
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Standard technical paths
BASE_DATA_PATH = "/mnt/pfs/loongflow/devmachine/h20-09/evolux/output/mlebench/iwildcam-2020-fgvc7/prepared/public"
OUTPUT_DATA_PATH = "output/9508c267-92c9-4fd0-91c8-90efc0fba263/1/executor/output"

# Logger setup
logger = logging.getLogger(__name__)

# Task-adaptive type definitions
X = Any      # Will be WildCamDataset (PyTorch Dataset)
y = pd.Series # Target vector type

class WildCamDataset(torch.utils.data.Dataset):
    """
    A custom PyTorch Dataset for iWildCam 2020 that implements 
    MegaDetector-based cropping and robust image augmentations.
    """
    def __init__(self, X_df: pd.DataFrame, y_series: pd.Series = None, transform: transforms.Compose = None):
        self.X_df = X_df.reset_index(drop=True)
        self.y = y_series.reset_index(drop=True) if y_series is not None else None
        self.transform = transform
        self.base_path = BASE_DATA_PATH

    def __len__(self) -> int:
        return len(self.X_df)

    def __getitem__(self, idx: int) -> Any:
        try:
            row = self.X_df.iloc[idx]
            img_path = os.path.join(self.base_path, row['file_name'])
            img = Image.open(img_path).convert('RGB')
            
            # Step 1: MegaDetector-based Cropping (Union of all boxes)
            bboxes = row.get('bboxes', [])
            if bboxes and isinstance(bboxes, list):
                # BBox format: [ymin, xmin, ymax, xmax] (relative)
                # Filter for union calculation (spec mentioned confidence > 0.1, 
                # but load_data only provides the bbox list itself)
                ymin = min(float(b[0]) for b in bboxes)
                xmin = min(float(b[1]) for b in bboxes)
                ymax = max(float(b[2]) for b in bboxes)
                xmax = max(float(b[3]) for b in bboxes)
                
                # Calculate dimensions for padding
                w_box = xmax - xmin
                h_box = ymax - ymin
                
                # Add 10% relative padding as per specification
                xmin = max(0.0, xmin - 0.1 * w_box)
                ymin = max(0.0, ymin - 0.1 * h_box)
                xmax = min(1.0, xmax + 0.1 * w_box)
                ymax = min(1.0, ymax + 0.1 * h_box)
                
                # Convert relative to absolute coordinates
                W_img, H_img = img.size
                left, top, right, bottom = xmin * W_img, ymin * H_img, xmax * W_img, ymax * H_img
                
                # Apply crop if coordinates define a valid area
                if right > left and bottom > top:
                    img = img.crop((left, top, right, bottom))
            
            # Step 2: Apply Augmentations and Normalization
            if self.transform:
                img = self.transform(img)
            else:
                # Default resizing if no transform is provided
                img = transforms.Compose([
                    transforms.Resize((384, 384)),
                    transforms.ToTensor()
                ])(img)
                
        except Exception as e:
            # MANDATORY: Wrap entire pipeline in try-except. 
            # Fallback to black tensor on any failure.
            logger.warning(f"Failed to process image at index {idx}: {e}")
            img = torch.zeros((3, 384, 384))
        
        # Return image and label if available
        if self.y is not None:
            label = self.y.iloc[idx]
            return img, torch.tensor(label, dtype=torch.long)
        
        return img

def preprocess(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame
) -> Tuple[X, y, X, y, X]:
    """
    Transforms raw data into model-ready Dataset format.
    Implementation focuses on spatial cropping and signal enhancement.
    """
    print("Starting iWildCam preprocessing pipeline...")

    # ImageNet normalization parameters
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )

    # Strategy: RandAugment (N=2, M=9) for training
    train_transform = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.RandAugment(num_ops=2, magnitude=9),
        transforms.ToTensor(),
        normalize
    ])

    # Validation and Test: Resize and Normalization only
    val_test_transform = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        normalize
    ])

    # Instantiate Datasets
    # Mixup (alpha=0.4) and Cutmix (alpha=1.0) are batch-level augmentations 
    # intended for the training loop in the next stage.
    X_train_processed = WildCamDataset(X_train, y_train, transform=train_transform)
    X_val_processed = WildCamDataset(X_val, y_val, transform=val_test_transform)
    X_test_processed = WildCamDataset(X_test, None, transform=val_test_transform)

    # Verification: Ensure test set completeness
    if len(X_test_processed) != len(X_test):
        raise ValueError("Preprocessing error: X_test_processed sample count mismatch.")

    print(f"Preprocessing completed. Train size: {len(X_train_processed)}, Val size: {len(X_val_processed)}, Test size: {len(X_test_processed)}")

    return X_train_processed, y_train, X_val_processed, y_val, X_test_processed