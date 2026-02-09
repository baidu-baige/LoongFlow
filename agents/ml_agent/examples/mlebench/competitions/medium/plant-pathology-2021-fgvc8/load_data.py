import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MultiLabelBinarizer
from typing import Tuple, Any

BASE_DATA_PATH = "/mnt/pfs/loongflow/devmachine/h20-01/evolux/output/mlebench/plant-pathology-2021-fgvc8/prepared/public"
OUTPUT_DATA_PATH = "output/e81e4df9-fbfb-4465-8b24-4be8ee1f51f4/1/executor/output"

# Define concrete types for this task
X = pd.Series    # Series of full image paths
y = np.ndarray  # Binary matrix (N, 6)
Ids = pd.Series  # Series of image filenames (e.g., "85f8cb619c66b863.jpg")

def load_data(validation_mode: bool = False) -> Tuple[X, y, X, Ids]:
    """
    Loads and prepares the datasets for the Plant Pathology 2021 challenge.
    Converts space-delimited labels into binary multi-label format and constructs image paths.
    """
    print(f"Loading data from {BASE_DATA_PATH}...")
    
    # Define paths
    train_csv_path = os.path.join(BASE_DATA_PATH, "train.csv")
    test_csv_path = os.path.join(BASE_DATA_PATH, "sample_submission.csv")
    train_img_dir = os.path.join(BASE_DATA_PATH, "train_images")
    test_img_dir = os.path.join(BASE_DATA_PATH, "test_images")

    # Verify critical files
    if not os.path.exists(train_csv_path):
        raise FileNotFoundError(f"Missing train metadata: {train_csv_path}")
    if not os.path.exists(test_csv_path):
        raise FileNotFoundError(f"Missing test metadata: {test_csv_path}")

    # Load metadata
    train_df = pd.read_csv(train_csv_path)
    test_df = pd.read_csv(test_csv_path)

    # Verify a sample image to ensure directory structure is correct
    sample_img = train_df['image'].iloc[0]
    if not os.path.exists(os.path.join(train_img_dir, sample_img)):
        raise FileNotFoundError(f"Image directory mismatch: {os.path.join(train_img_dir, sample_img)} not found.")

    # Multi-label binarization
    # Specification classes: ['scab', 'healthy', 'frog_eye_leaf_spot', 'rust', 'complex', 'powdery_mildew']
    target_classes = ['scab', 'healthy', 'frog_eye_leaf_spot', 'rust', 'complex', 'powdery_mildew']
    mlb = MultiLabelBinarizer(classes=target_classes)
    
    # Split labels by space and transform
    labels_split = train_df['labels'].str.split(' ')
    y_train = mlb.fit_transform(labels_split).astype(np.float32)
    
    # Construct full paths for training images
    X_train = train_df['image'].apply(lambda x: os.path.join(train_img_dir, x))
    
    # Prepare test data
    # In this competition, test_images is where the actual test images reside.
    # sample_submission.csv provides the IDs.
    X_test = test_df['image'].apply(lambda x: os.path.join(test_img_dir, x))
    test_ids = test_df['image']

    # Validation mode subsetting
    if validation_mode:
        print("Validation mode enabled: subsetting data to 200 rows.")
        limit = 200
        X_train = X_train.iloc[:limit]
        y_train = y_train[:limit]
        X_test = X_test.iloc[:limit]
        test_ids = test_ids.iloc[:limit]

    print(f"Data loading complete. Train: {len(X_train)} samples, Test: {len(X_test)} samples.")
    print(f"Target classes: {target_classes}")

    return X_train, y_train, X_test, test_ids