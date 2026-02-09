import os
import pandas as pd
import numpy as np
from typing import Tuple, List

BASE_DATA_PATH = "/mnt/pfs/loongflow/devmachine/h20-12/evolux/output/mlebench/plant-pathology-2020-fgvc7/prepared/public"
OUTPUT_DATA_PATH = "output/fc83b0b0-0bd1-41b2-8cd2-ac42c45cd457/2/executor/output"

# Task-adaptive type definitions
X = List[str]      # Paths to image files
y = np.ndarray    # Multi-label target matrix (N, 4)
Ids = np.ndarray  # Array of image_ids for test set

def load_data(validation_mode: bool = False) -> Tuple[X, y, X, Ids]:
    """
    Loads and prepares the datasets required for the Plant Pathology 2020 task.
    Uses pre-resized 512x512 images as per technical specification.

    Args:
        validation_mode: If True, returns a subset of at most 200 rows.

    Returns:
        Tuple[X, y, X, Ids]: 
            X_train_paths (List[str]): List of absolute paths to training images.
            y_train (np.ndarray): One-hot/Multi-label encoded targets for training.
            X_test_paths (List[str]): List of absolute paths to test images.
            test_ids (np.ndarray): Image IDs for the test set.
    """
    print("Stage 1: Loading data...")
    
    # Define paths
    train_csv_path = os.path.join(BASE_DATA_PATH, "train.csv")
    test_csv_path = os.path.join(BASE_DATA_PATH, "test.csv")
    img_dir = os.path.join(BASE_DATA_PATH, "resized_512")
    
    # Verify critical file existence
    if not os.path.exists(train_csv_path):
        raise FileNotFoundError(f"Required file train.csv not found at {train_csv_path}")
    if not os.path.exists(test_csv_path):
        raise FileNotFoundError(f"Required file test.csv not found at {test_csv_path}")
    if not os.path.exists(img_dir):
        raise FileNotFoundError(f"Resized image directory not found at {img_dir}. Ensure data preparation is complete.")

    # Load metadata
    train_df = pd.read_csv(train_csv_path)
    test_df = pd.read_csv(test_csv_path)
    
    # Identify target columns (excluding image_id)
    # Expected: ['healthy', 'multiple_diseases', 'rust', 'scab']
    target_cols = [c for c in train_df.columns if c != 'image_id']
    
    # Apply validation subsetting
    if validation_mode:
        print(f"Validation mode active: Subsetting data to 200 samples.")
        train_df = train_df.head(200)
        test_df = test_df.head(200)
    
    # Construct full paths for images
    # The resized_512 directory contains images named {image_id}.jpg
    X_train_paths = [os.path.join(img_dir, f"{idx}.jpg") for idx in train_df['image_id']]
    y_train = train_df[target_cols].values.astype(np.float32)
    
    X_test_paths = [os.path.join(img_dir, f"{idx}.jpg") for idx in test_df['image_id']]
    test_ids = test_df['image_id'].values
    
    # Final sanity checks
    if len(X_train_paths) == 0:
        raise ValueError("Training set is empty.")
    if len(X_test_paths) == 0:
        raise ValueError("Test set is empty.")
    if len(X_train_paths) != len(y_train):
        raise ValueError(f"Train features ({len(X_train_paths)}) and targets ({len(y_train)}) mismatch.")
    if len(X_test_paths) != len(test_ids):
        raise ValueError(f"Test features ({len(X_test_paths)}) and IDs ({len(test_ids)}) mismatch.")
        
    # Verify first path exists to ensure directory mapping is correct
    if not os.path.exists(X_train_paths[0]):
        raise FileNotFoundError(f"Could not find image at {X_train_paths[0]}. Check directory structure.")

    print(f"Successfully loaded {len(X_train_paths)} training samples and {len(X_test_paths)} test samples.")
    print(f"Target columns: {target_cols}")

    return X_train_paths, y_train, X_test_paths, test_ids