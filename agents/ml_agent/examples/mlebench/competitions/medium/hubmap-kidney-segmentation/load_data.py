import os
import pandas as pd
from typing import Tuple, Any
import tifffile

# Task-adaptive type definitions
X = pd.DataFrame  # Feature matrix: Metadata and image paths
y = pd.Series     # Target vector: RLE strings
Ids = pd.Series   # Identifier type: Image IDs

BASE_DATA_PATH = "/mnt/pfs/loongflow/devmachine/h20-01/evolux/output/mlebench/hubmap-kidney-segmentation/prepared/public"
OUTPUT_DATA_PATH = "output/dfecbcec-c1a9-41b2-a54c-960bf09a6314/1/executor/output"

def load_data(validation_mode: bool = False) -> Tuple[X, y, X, Ids]:
    """
    Loads and prepares the datasets for the HuBMAP Kidney Segmentation task.
    Uses lazy-loading strategy by providing paths to massive TIFF files.
    """
    print("Stage 1: Loading data metadata and paths...")

    # Define paths to source files
    train_csv_path = os.path.join(BASE_DATA_PATH, "train.csv")
    info_csv_path = os.path.join(BASE_DATA_PATH, "HuBMAP-20-dataset_information.csv")
    train_dir = os.path.join(BASE_DATA_PATH, "train")
    test_dir = os.path.join(BASE_DATA_PATH, "test")

    # Load core files
    if not os.path.exists(train_csv_path):
        raise FileNotFoundError(f"Required file not found: {train_csv_path}")
    if not os.path.exists(info_csv_path):
        raise FileNotFoundError(f"Required file not found: {info_csv_path}")

    train_df = pd.read_csv(train_csv_path)
    info_df = pd.read_csv(info_csv_path)

    # Standardize IDs in info_df (strip .tiff from image_file)
    info_df['id'] = info_df['image_file'].apply(lambda x: x.replace('.tiff', ''))

    # Prepare Training Data
    print("Preparing training metadata...")
    X_train = train_df.merge(info_df, on='id', how='left')
    X_train['img_path'] = X_train['id'].apply(lambda x: os.path.join(train_dir, f"{x}.tiff"))
    
    # Verify training image paths
    for path in X_train['img_path']:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Training image missing at path: {path}")

    y_train = X_train['encoding']
    X_train = X_train.drop(columns=['encoding'])

    # Prepare Test Data
    print("Preparing test metadata...")
    # Identify test files in the directory
    test_files = [f for f in os.listdir(test_dir) if f.endswith('.tiff')]
    if not test_files:
        # Fallback to sample_submission if directory scan fails to find .tiff files
        sample_sub_path = os.path.join(BASE_DATA_PATH, "sample_submission.csv")
        if os.path.exists(sample_sub_path):
            test_ids_list = pd.read_csv(sample_sub_path)['id'].tolist()
        else:
            raise FileNotFoundError("No test images found in 'test/' and no sample_submission.csv available.")
    else:
        test_ids_list = [f.replace('.tiff', '') for f in test_files]

    X_test = pd.DataFrame({'id': test_ids_list})
    X_test = X_test.merge(info_df, on='id', how='left')
    X_test['img_path'] = X_test['id'].apply(lambda x: os.path.join(test_dir, f"{x}.tiff"))

    # Verify test image paths
    for path in X_test['img_path']:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Test image missing at path: {path}")

    test_ids = X_test['id']

    # Apply validation_mode subsetting
    if validation_mode:
        print(f"Validation mode active: subsetting data to max 200 rows.")
        X_train = X_train.head(200).reset_index(drop=True)
        y_train = y_train.head(200).reset_index(drop=True)
        X_test = X_test.head(200).reset_index(drop=True)
        test_ids = test_ids.head(200).reset_index(drop=True)

    # Final cleanup and reporting
    print(f"Data loading complete.")
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    
    return X_train, y_train, X_test, test_ids