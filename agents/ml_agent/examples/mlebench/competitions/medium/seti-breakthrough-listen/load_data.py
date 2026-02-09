import os
import pandas as pd
from typing import Tuple, Any

BASE_DATA_PATH = "/mnt/pfs/loongflow/devmachine/h20-09/evolux/output/mlebench/seti-breakthrough-listen/prepared/public"
OUTPUT_DATA_PATH = "output/a429c40e-fe12-455c-8b05-ca9d732aabeb/1/executor/output"

# Task-adaptive type definitions
X = pd.DataFrame  # Feature matrix type (containing metadata and file paths)
y = pd.Series     # Target vector type
Ids = pd.Series   # Identifier type for output alignment

def get_file_path(row_id: str, folder: str) -> str:
    """
    Constructs the file path for a given ID. 
    The competition uses a nested directory structure based on the first character of the ID.
    Example: train/0/00034abb3629.npy
    """
    return os.path.join(BASE_DATA_PATH, folder, row_id[0], f"{row_id}.npy")

def load_data(validation_mode: bool = False) -> Tuple[X, y, X, Ids]:
    """
    Loads and prepares the datasets required for the SETI Breakthrough Listen task.
    Returns file path mappings rather than raw arrays to stay memory efficient.
    """
    print(f"Loading data with validation_mode={validation_mode}...")

    # Define paths to labels and sample submission
    train_labels_path = os.path.join(BASE_DATA_PATH, "train_labels.csv")
    sample_sub_path = os.path.join(BASE_DATA_PATH, "sample_submission.csv")

    # Step 1: Load CSV files
    train_df = pd.read_csv(train_labels_path)
    test_df = pd.read_csv(sample_sub_path)

    # Step 2: Verify path structure by sampling
    # Check if the nested structure (id[0]) exists, otherwise assume flat structure
    sample_id = train_df['id'].iloc[0]
    nested_path = get_file_path(sample_id, "train")
    flat_path = os.path.join(BASE_DATA_PATH, "train", f"{sample_id}.npy")

    if os.path.exists(nested_path):
        folder_structure = "nested"
    elif os.path.exists(flat_path):
        folder_structure = "flat"
    else:
        # If neither exists, let it fail early to propagate error
        raise FileNotFoundError(f"Could not find .npy files in {BASE_DATA_PATH}/train/ using nested or flat structure.")

    print(f"Detected folder structure: {folder_structure}")

    # Step 3: Construct full paths for all samples
    if folder_structure == "nested":
        train_df['filepath'] = train_df['id'].apply(lambda x: get_file_path(x, "train"))
        test_df['filepath'] = test_df['id'].apply(lambda x: get_file_path(x, "test"))
    else:
        train_df['filepath'] = train_df['id'].apply(lambda x: os.path.join(BASE_DATA_PATH, "train", f"{x}.npy"))
        test_df['filepath'] = test_df['id'].apply(lambda x: os.path.join(BASE_DATA_PATH, "test", f"{x}.npy"))

    # Step 4: Subset for validation mode if required
    if validation_mode:
        print("Subsetting data for validation mode...")
        # Stratified representative sampling if possible, otherwise simple head
        if 'target' in train_df.columns:
            train_subset = train_df.groupby('target', group_keys=False).apply(lambda x: x.head(100))
            if len(train_subset) > 200:
                train_subset = train_subset.head(200)
            train_df = train_subset
        else:
            train_df = train_df.head(200)
        
        test_df = test_df.head(200)

    # Step 5: Structure into return format
    # X_train: df with ['id', 'filepath']
    # y_train: target series
    # X_test: df with ['id', 'filepath']
    # test_ids: id series
    
    X_train = train_df[['id', 'filepath']]
    y_train = train_df['target']
    X_test = test_df[['id', 'filepath']]
    test_ids = test_df['id']

    print(f"Data loading complete. Train size: {len(X_train)}, Test size: {len(X_test)}")
    
    # Final check for row alignment
    assert len(X_train) == len(y_train), "X_train and y_train alignment mismatch"
    assert len(X_test) == len(test_ids), "X_test and test_ids alignment mismatch"

    return X_train, y_train, X_test, test_ids