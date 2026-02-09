from typing import Tuple, Any
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from sklearn.preprocessing import StandardScaler

BASE_DATA_PATH = "/mnt/pfs/loongflow/devmachine/h20-09/evolux/output/mlebench/iwildcam-2019-fgvc6/prepared/public"
OUTPUT_DATA_PATH = "output/939424a9-5a07-4c99-9f56-709187b5b05c/1/executor/output"

# Task-adaptive type definitions
# X is a Dataset object that lazily loads images and metadata tensors
# y is a pd.Series containing the target labels
X = Any
y = pd.Series

class IWildCamDataset(Dataset):
    """
    A robust dataset class for iWildCam 2019 that handles image loading and metadata.
    Implements lazy loading to efficiently manage the 440GB RAM constraint while 
    processing ~26GB of raw images.
    """
    def __init__(self, df: pd.DataFrame, meta_values: np.ndarray, transform: transforms.Compose):
        """
        Args:
            df (pd.DataFrame): Dataframe containing 'image_path'.
            meta_values (np.ndarray): Pre-scaled metadata array of shape (N, 2).
            transform (transforms.Compose): Torchvision transforms to apply to images.
        """
        self.df = df.reset_index(drop=True)
        self.meta_values = meta_values.astype(np.float32)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path = self.df.iloc[idx]['image_path']
        
        # Robust Loading: Mitigate image corruption (<0.1% of images)
        try:
            # Use PIL.Image.open and convert to RGB
            image = Image.open(img_path).convert('RGB')
        except Exception:
            # For failed loads, use a blank placeholder (black image) as per specification
            image = Image.new('RGB', (384, 384), (0, 0, 0))
        
        # Apply image transformations (Resize/Crop, Augmentations, ToTensor, Normalize)
        if self.transform:
            image = self.transform(image)
        
        # Numerical Metadata: Return as a tensor of shape (2,)
        meta = torch.tensor(self.meta_values[idx], dtype=torch.float32)
        
        return image, meta

def preprocess(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame
) -> Tuple[X, y, X, y, X]:
    """
    Transforms raw data into model-ready format for a single fold/split.
    Handles image augmentation for domain invariance and metadata scaling.
    """
    print("Starting preprocessing stage...")

    # Step 1: Define Image Transformations
    # ImageNet normalization statistics
    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]

    # Training augmentations to mitigate regional domain shift (lighting, night-vision, background)
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(384, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=norm_mean, std=norm_std)
    ])

    # Validation and Test transformations (deterministic resize and normalization)
    val_test_transform = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize(mean=norm_mean, std=norm_std)
    ])

    # Step 2: Metadata Scaling (StandardScale for seq_num_frames and frame_num)
    meta_cols = ['seq_num_frames', 'frame_num']
    print(f"Scaling metadata columns: {meta_cols}")
    
    scaler = StandardScaler()
    
    # Fit scaler on training data ONLY to prevent information leakage
    scaler.fit(X_train[meta_cols])
    
    # Transform metadata for all sets
    X_train_meta = scaler.transform(X_train[meta_cols])
    X_val_meta = scaler.transform(X_val[meta_cols])
    X_test_meta = scaler.transform(X_test[meta_cols])

    # Step 3: Construct Dataset objects
    # These objects are returned as X_train_processed, etc.
    # They are "model-ready" as they produce (Image Tensor, Meta Tensor) upon indexing.
    print("Constructing Dataset objects...")
    X_train_processed = IWildCamDataset(X_train, X_train_meta, train_transform)
    X_val_processed = IWildCamDataset(X_val, X_val_meta, val_test_transform)
    X_test_processed = IWildCamDataset(X_test, X_test_meta, val_test_transform)

    # Step 4: Ensure target consistency
    # category_id is already in integer format (0-22)
    y_train_processed = y_train.copy()
    y_val_processed = y_val.copy()

    # Step 5: Verification of output integrity
    # Ensure no NaN or Inf in scaled metadata
    for meta_arr, name in zip([X_train_meta, X_val_meta, X_test_meta], ["train", "val", "test"]):
        if not np.isfinite(meta_arr).all():
            raise ValueError(f"Metadata in {name} set contains NaN or Inf after scaling.")

    print(f"Preprocessing complete. Train samples: {len(X_train_processed)}, Test samples: {len(X_test_processed)}")
    
    return X_train_processed, y_train_processed, X_val_processed, y_val_processed, X_test_processed