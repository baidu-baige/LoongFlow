import pandas as pd
import os
from typing import Tuple

BASE_DATA_PATH = "/mnt/pfs/loongflow/devmachine/h20-03/evolux/output/mlebench/jigsaw-toxic-comment-classification-challenge/prepared/public"
OUTPUT_DATA_PATH = "output/4d08636e-bf37-40e0-b9d7-8ffb77d57ea2/1/executor/output"

# Task-adaptive type definitions
X = pd.DataFrame  # Feature matrix: contains 'comment_text'
y = pd.DataFrame  # Target vector: contains 6 toxicity labels
Ids = pd.Series   # Identifier type: contains 'id' for test set

def load_data(validation_mode: bool = False) -> Tuple[X, y, X, Ids]:
    """
    Loads and prepares the datasets for the Toxic Comment Classification Challenge.

    Args:
        validation_mode: If True, returns a small subset (200 rows) for testing.

    Returns:
        X_train, y_train, X_test, test_ids
    """
    print(f"Starting data loading. Validation mode: {validation_mode}")

    # Define paths
    train_csv_path = os.path.join(BASE_DATA_PATH, "train.csv")
    test_csv_path = os.path.join(BASE_DATA_PATH, "test.csv")

    # Verify files exist
    for path in [train_csv_path, test_csv_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required data file not found: {path}")

    # Define column structures
    target_columns = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    train_use_cols = ['comment_text'] + target_columns
    test_use_cols = ['id', 'comment_text']

    # Load parameters
    load_params_train = {
        "filepath_or_buffer": train_csv_path,
        "usecols": train_use_cols,
        "dtype": {col: 'int8' for col in target_columns},
        "engine": 'c'
    }
    load_params_test = {
        "filepath_or_buffer": test_csv_path,
        "usecols": test_use_cols,
        "engine": 'c'
    }

    if validation_mode:
        load_params_train["nrows"] = 200
        load_params_test["nrows"] = 200

    # Load DataFrames
    print("Reading train.csv...")
    train_df = pd.read_csv(**load_params_train)
    print("Reading test.csv...")
    test_df = pd.read_csv(**load_params_test)

    # Clean text data: ensure string type and fill missing
    print("Pre-processing text columns...")
    train_df['comment_text'] = train_df['comment_text'].fillna('').astype(str)
    test_df['comment_text'] = test_df['comment_text'].fillna('').astype(str)

    # Structure outputs
    X_train = train_df[['comment_text']]
    y_train = train_df[target_columns]
    X_test = test_df[['comment_text']]
    test_ids = test_df['id']

    # Final sanity check
    assert len(X_train) == len(y_train), "Train features and targets mismatch"
    assert len(X_test) == len(test_ids), "Test features and IDs mismatch"
    assert not X_train.empty and not y_train.empty and not X_test.empty and not test_ids.empty, "Empty data returned"

    print(f"Data loading complete. Train size: {len(X_train)}, Test size: {len(X_test)}")
    
    return X_train, y_train, X_test, test_ids