from typing import Tuple, Any
import pandas as pd
import os

BASE_DATA_PATH = "/root/workspace/evolux_ml/output/mlebench/aptos2019-blindness-detection/prepared/public"
OUTPUT_DATA_PATH = "output/16326c74-72ad-4b59-ad28-cc76a3d9d373/5/executor/output"

# Task-adaptive type definitions
X = pd.Series   # Series containing full string paths to image files
y = pd.Series  # Series containing diagnosis labels as float32
Ids = pd.Series  # Series containing id_code strings for alignment

def load_data(validation_mode: bool = False) -> Tuple[X, y, X, Ids]:
    """
    Loads and prepares the datasets for the APTOS 2019 Blindness Detection task.
    Maps metadata to full image file paths and prepares targets for regression-based training.
    """
    
    # Define paths for metadata files
    train_csv_path = os.path.join(BASE_DATA_PATH, "train.csv")
    test_csv_path = os.path.join(BASE_DATA_PATH, "test.csv")
    
    # Load metadata
    # Ensure errors propagate if CSVs are missing
    train_df = pd.read_csv(train_csv_path)
    test_df = pd.read_csv(test_csv_path)
    
    # Handle validation mode subsetting (<= 200 rows)
    if validation_mode:
        train_df = train_df.head(200)
        test_df = test_df.head(200)
        
    # Define image directories
    train_img_dir = os.path.join(BASE_DATA_PATH, "train_images")
    test_img_dir = os.path.join(BASE_DATA_PATH, "test_images")
    
    # Construct full image paths using id_code and .png extension
    # This logic is adapted from the parent to ensure robust mapping
    train_df['image_path'] = train_df['id_code'].apply(
        lambda x: os.path.join(train_img_dir, f"{x}.png")
    )
    test_df['image_path'] = test_df['id_code'].apply(
        lambda x: os.path.join(test_img_dir, f"{x}.png")
    )
    
    # Strict Verification: Verify that every mapped image file actually exists.
    # We let the FileNotFoundError propagate immediately to ensure data integrity.
    for path in train_df['image_path']:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Critical error: Training image not found at {path}")
            
    for path in test_df['image_path']:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Critical error: Testing image not found at {path}")
            
    # Structure returns
    # y are converted to float32 to facilitate regression-based approaches,
    # which is a standard strategy for optimizing Quadratic Weighted Kappa (QWK).
    x_train = train_df['image_path']
    y_train = train_df['diagnosis'].astype('float32')
    
    x_test = test_df['image_path']
    test_ids = test_df['id_code']
    
    return x_train, y_train, x_test, test_ids