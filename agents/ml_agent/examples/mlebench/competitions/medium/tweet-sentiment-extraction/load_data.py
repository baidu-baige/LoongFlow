import pandas as pd
import os
from typing import Tuple

# Task-adaptive type definitions
X = pd.DataFrame    # Feature matrix containing 'text' and 'sentiment'
y = pd.Series       # Target vector containing 'selected_text'
Ids = pd.Series     # Identifier vector containing 'textID'

BASE_DATA_PATH = "/mnt/pfs/loongflow/devmachine/1-04/evolux/output/mlebench/tweet-sentiment-extraction/prepared/public"
OUTPUT_DATA_PATH = "output/1cc1b6ac-193c-4f3f-a388-380b832f53e8/5/executor/output"

def load_data(validation_mode: bool = False) -> Tuple[X, y, X, Ids]:
    """
    Loads and prepares the datasets required for the tweet sentiment extraction task.
    Ensures structural integrity and preserves whitespace for character-level alignment.

    Args:
        validation_mode: If True, returns a subset of at most 200 rows for quick testing.

    Returns:
        Tuple[X, y, X, Ids]: (X_train, y_train, X_test, test_ids)
    """
    train_path = os.path.join(BASE_DATA_PATH, "train.csv")
    test_path = os.path.join(BASE_DATA_PATH, "test.csv")

    # Step 0: Ensure data readiness
    # Check if files exist to avoid silent failures; propagate errors if missing
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Critical training file not found at: {train_path}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Critical test file not found at: {test_path}")

    print(f"Loading raw data from {BASE_DATA_PATH}...")

    # Step 1: Load data from sources
    # Technical Specification: Use pd.read_csv with keep_default_na=False and dtype=str.
    # This prevents 'nan' strings or empty strings from being converted to float NaNs,
    # which is critical for text processing and maintaining substring indices.
    try:
        train_df = pd.read_csv(train_path, keep_default_na=False, dtype=str)
        test_df = pd.read_csv(test_path, keep_default_na=False, dtype=str)
    except Exception as e:
        print(f"Failed to load CSV files: {e}")
        raise

    # Step 2: Structure data into required return format
    # Verify presence of essential columns identified in Task Description
    required_train_cols = ['text', 'sentiment', 'selected_text']
    required_test_cols = ['text', 'sentiment', 'textID']

    if not all(col in train_df.columns for col in required_train_cols):
        missing = [c for c in required_train_cols if c not in train_df.columns]
        raise ValueError(f"Training data is missing required columns: {missing}")
    
    if not all(col in test_df.columns for col in required_test_cols):
        missing = [c for c in required_test_cols if c not in test_df.columns]
        raise ValueError(f"Test data is missing required columns: {missing}")

    # Step 3: Apply validation_mode subsetting if enabled
    # Preparation (the load above) is already "full" before subsetting.
    if validation_mode:
        print("Validation mode: Active. Subsetting data to at most 200 samples.")
        train_df = train_df.head(200)
        test_df = test_df.head(200)

    # Separation of features, targets, and IDs
    X_train = train_df[['text', 'sentiment']]
    y_train = train_df['selected_text']
    
    X_test = test_df[['text', 'sentiment']]
    test_ids = test_df['textID']

    # Step 4: Final verification of row alignment and non-emptiness
    if X_train.empty or X_test.empty:
        raise ValueError("Loaded datasets must not be empty.")
    
    if len(X_train) != len(y_train):
        raise ValueError(f"Training features ({len(X_train)}) and targets ({len(y_train)}) misalignment.")
    
    if len(X_test) != len(test_ids):
        raise ValueError(f"Test features ({len(X_test)}) and IDs ({len(test_ids)}) misalignment.")

    print(f"Data loading successful. Train size: {len(X_train)}, Test size: {len(X_test)}")
    
    return X_train, y_train, X_test, test_ids