import os
import pandas as pd
from typing import Tuple

BASE_DATA_PATH = "/mnt/pfs/loongflow/devmachine/2-05/evolux/output/mlebench/random-acts-of-pizza/prepared/public"
OUTPUT_DATA_PATH = "output/f2dbb22d-a0cb-4add-aa87-f2c6b1a4b76f/77/executor/output"

# Task-adaptive type definitions
X = pd.DataFrame    # Feature matrix type
y = pd.Series       # Target vector type
Ids = pd.Series     # Identifier type for output alignment

def load_data(validation_mode: bool = False) -> Tuple[X, y, X, Ids]:
    """
    Loads and prepares the datasets for the Random Acts of Pizza task.
    Filters features to include only those available at request time based on the test set schema.

    Args:
        validation_mode: Controls the data loading behavior.
            - False (default): Load the complete dataset.
            - True: Return a subset (at most 200 rows) for quick validation.
    Returns:
        Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]: 
        X_train, y_train, X_test, test_ids
    """
    # Step 0: Define paths and verify existence
    train_path = os.path.join(BASE_DATA_PATH, "train.json")
    test_path = os.path.join(BASE_DATA_PATH, "test.json")
    
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Critical Error: Training data not found at {train_path}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Critical Error: Test data not found at {test_path}")

    # Step 1: Load data from sources
    print(f"Loading data from {BASE_DATA_PATH}...")
    try:
        # JSON files are ~12MB total; pandas handles this efficiently in memory
        train_df = pd.read_json(train_path)
        test_df = pd.read_json(test_path)
    except Exception as e:
        print(f"Failed to load JSON files: {e}")
        raise

    # Step 2: Structure data into required return format
    # The target column as per dataset description
    target_col = 'requester_received_pizza'
    id_col = 'request_id'

    if target_col not in train_df.columns:
        raise KeyError(f"Target column '{target_col}' not found in training data.")

    # Restrict features to those available in the test set to prevent data leakage.
    # The test set only contains fields available at the time of posting.
    feature_cols = [col for col in test_df.columns if col != id_col]
    
    # Verify feature alignment
    missing_cols = [col for col in feature_cols if col not in train_df.columns]
    if missing_cols:
        raise KeyError(f"Critical Error: Feature columns {missing_cols} missing from training data.")

    # Extract features, target, and IDs
    X_train = train_df[feature_cols].copy()
    y_train = train_df[target_col].copy()
    X_test = test_df[feature_cols].copy()
    test_ids = test_df[id_col].copy()

    # Set 'request_id' as index for the feature dataframes for easier tracking
    X_train.index = train_df[id_col].values
    X_test.index = test_df[id_col].values

    # Step 3: Apply validation_mode subsetting if enabled
    if validation_mode:
        print("Validation mode enabled: subsetting to 200 samples.")
        limit = 200
        X_train = X_train.head(limit)
        y_train = y_train.head(limit)
        X_test = X_test.head(limit)
        test_ids = test_ids.head(limit)

    # Step 4: Final verification and return
    if X_train.empty or y_train.empty or X_test.empty or test_ids.empty:
        raise ValueError("Data loading produced empty datasets.")
    
    if len(X_train) != len(y_train):
        raise ValueError(f"Training feature/target mismatch: {len(X_train)} vs {len(y_train)}")
        
    if len(X_test) != len(test_ids):
        raise ValueError(f"Test feature/ID mismatch: {len(X_test)} vs {len(test_ids)}")

    print(f"Data load completed successfully.")
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}, test_ids shape: {test_ids.shape}")
    
    return X_train, y_train, X_test, test_ids