import os
import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from typing import Tuple, Any
import concurrent.futures
from tqdm import tqdm

# Task-adaptive type definitions
# X is defined as a Dataset to satisfy the technical requirement for PyTorch Datasets 
# while ensuring row alignment (len(X) == len(y)) for downstream splitting.
X = Dataset  
y = pd.Series    
Ids = pd.Series  

BASE_DATA_PATH = "/mnt/pfs/loongflow/devmachine/2-05/evolux/output/mlebench/whale-categorization-playground/prepared/public"

class WhaleDataset(Dataset):
    """
    Optimized PyTorch Dataset for individual whale identification.
    Uses RAM-caching (uint8) to maximize throughput and minimize I/O bottlenecks.
    Preserves fine-grained details by using high-quality resizing.
    """
    def __init__(self, df: pd.DataFrame, image_dir: str, transform: transforms.Compose = None):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform
        self.image_names = self.df['Image'].values
        
        # Pre-allocate memory for all images in uint8 to optimize RAM usage (approx 8.7 GB for full dataset)
        num_samples = len(self.df)
        self.images = torch.zeros((num_samples, 3, 384, 768), dtype=torch.uint8)
        
        print(f"Pre-loading {num_samples} images from {image_dir} into RAM...")
        
        def load_and_resize(idx):
            img_path = os.path.join(self.image_dir, self.image_names[idx])
            try:
                img = Image.open(img_path).convert('RGB')
                # Resize to 768x384 as per specification using Lanczos for detail preservation
                img = img.resize((768, 384), Image.LANCZOS)
                # Convert to tensor (C, H, W)
                return idx, torch.from_numpy(np.array(img)).permute(2, 0, 1)
            except Exception as e:
                raise RuntimeError(f"Failed to process image {img_path}: {e}")

        # Maximize utilization using 36 available cores for parallel processing
        with concurrent.futures.ThreadPoolExecutor(max_workers=36) as executor:
            futures = [executor.submit(load_and_resize, i) for i in range(num_samples)]
            for future in tqdm(concurrent.futures.as_completed(futures), total=num_samples, desc="RAM Caching"):
                idx, img_tensor = future.result()
                self.images[idx] = img_tensor

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> torch.Tensor:
        # Handle single index access
        # Convert uint8 RAM storage to float32 [0, 1] for model consumption
        img = self.images[idx].float() / 255.0
        if self.transform:
            img = self.transform(img)
        return img

def load_data(validation_mode: bool = False) -> Tuple[X, y, X, Ids]:
    """
    Loads and prepares the training and test datasets.
    
    Args:
        validation_mode: If True, returns at most 200 samples for quick testing.
        
    Returns:
        X_train: WhaleDataset containing training images resized to 768x384.
        y_train: pd.Series of whale IDs for training.
        X_test: WhaleDataset containing test images resized to 768x384.
        test_ids: pd.Series of test image filenames for alignment.
    """
    BASE_PATH = "/mnt/pfs/loongflow/devmachine/2-05/evolux/output/mlebench/whale-categorization-playground/prepared/public"
    train_csv_path = os.path.join(BASE_PATH, "train.csv")
    sample_sub_path = os.path.join(BASE_PATH, "sample_submission.csv")
    
    # Step 1: Load metadata
    train_df_full = pd.read_csv(train_csv_path)
    test_df_full = pd.read_csv(sample_sub_path)
    
    # Step 2: Apply validation subsetting if requested
    if validation_mode:
        print("Validation mode enabled: Subsetting data to 200 rows.")
        train_df = train_df_full.head(200).copy()
        test_df = test_df_full.head(200).copy()
    else:
        train_df = train_df_full.copy()
        test_df = test_df_full.copy()

    # Step 3: Define normalization (ImageNet stats as per specification)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    # Step 4: Initialize Datasets
    # This triggers the pre-loading and resizing logic to ensure high-throughput in downstream stages
    print("Preparing Training Features...")
    X_train = WhaleDataset(
        df=train_df, 
        image_dir=os.path.join(BASE_PATH, "train"), 
        transform=normalize
    )
    y_train = train_df['Id']
    
    print("Preparing Test Features...")
    X_test = WhaleDataset(
        df=test_df, 
        image_dir=os.path.join(BASE_PATH, "test"), 
        transform=normalize
    )
    test_ids = test_df['Image']
    
    # Final alignment verification
    if len(X_train) != len(y_train):
        raise ValueError(f"Training alignment error: X_train has {len(X_train)} items, y_train has {len(y_train)} labels.")
    if len(X_test) != len(test_ids):
        raise ValueError(f"Test alignment error: X_test has {len(X_test)} items, test_ids has {len(test_ids)} identifiers.")

    print(f"Data loading complete. Train size: {len(X_train)}, Test size: {len(X_test)}")
    return X_train, y_train, X_test, test_ids