import os
import pandas as pd
from typing import Tuple

# Task-adaptive type definitions
X = pd.DataFrame      # Feature matrix type: DataFrame with columns ['id', 'path']
y = pd.Series         # Target vector type: Series of binary labels
Ids = pd.Series       # Identifier type: Series of test IDs

BASE_DATA_PATH = "/mnt/pfs/loongflow/devmachine/1-04/evolux/output/mlebench/histopathologic-cancer-detection/prepared/public"
OUTPUT_DATA_PATH = "output/cf83edc4-8764-4cf8-95a0-4f4a823260c7/2/executor/output"

def load_data(validation_mode: bool = False) -> Tuple[X, y, X, Ids]:
    """
    Loads and prepares the datasets required for the Histopathologic Cancer Detection task.
    Returns metadata DataFrames containing file paths and IDs to support efficient disk-streaming.
    """
    print(f"Initializing data loading from: {BASE_DATA_PATH}")

    # Define paths to source files
    train_labels_file = os.path.join(BASE_DATA_PATH, 'train_labels.csv')
    sample_sub_file = os.path.join(BASE_DATA_PATH, 'sample_submission.csv')
    train_dir = os.path.join(BASE_DATA_PATH, 'train')
    test_dir = os.path.join(BASE_DATA_PATH, 'test')

    # Ensure critical files and directories exist
    for path in [train_labels_file, sample_sub_file, train_dir, test_dir]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required data source missing: {path}")

    # Load labels metadata
    print("Reading train labels and sample submission...")
    train_df = pd.read_csv(train_labels_file)
    test_df = pd.read_csv(sample_sub_file)

    # Construct full absolute paths for each image ID
    # Note: filenames in train/test are {id}.tif
    train_df['path'] = train_df['id'].apply(lambda x: os.path.join(train_dir, f"{x}.tif"))
    test_df['path'] = test_df['id'].apply(lambda x: os.path.join(test_dir, f"{x}.tif"))

    # Verify path existence by sampling to prevent downstream I/O failures
    print("Verifying data integrity by sampling file paths...")
    for label, df in [("train", train_df), ("test", test_df)]:
        sample_check = df['path'].head(10).tolist()
        for p in sample_check:
            if not os.path.exists(p):
                raise FileNotFoundError(f"Data alignment error: File {p} not found in {label} directory.")

    # Apply validation_mode subsetting
    if validation_mode:
        print("Validation mode active: Subsetting data to 200 samples.")
        # Use fixed seed for deterministic subsetting in validation mode
        train_df = train_df.sample(n=min(200, len(train_df)), random_state=42).reset_index(drop=True)
        test_df = test_df.sample(n=min(200, len(test_df)), random_state=42).reset_index(drop=True)

    # Structure outputs according to specification
    X_train = train_df[['id', 'path']]
    y_train = train_df['label']
    X_test = test_df[['id', 'path']]
    test_ids = test_df['id']

    # Final validation of row alignment
    assert len(X_train) == len(y_train), "Mismatch between X_train and y_train lengths"
    assert len(X_test) == len(test_ids), "Mismatch between X_test and test_ids lengths"

    print(f"Data loading complete. Train samples: {len(X_train)}, Test samples: {len(X_test)}")
    
    return X_train, y_train, X_test, test_ids