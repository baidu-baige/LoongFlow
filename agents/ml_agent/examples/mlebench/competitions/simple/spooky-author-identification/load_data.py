import os
import pandas as pd
from typing import Tuple

# Task-adaptive type definitions
# X: Feature matrix containing the raw text snippets as a DataFrame.
# y: Target vector containing the categorical author labels.
# Ids: Identifier vector for the test set.
X = pd.DataFrame
y = pd.Series
Ids = pd.Series

BASE_DATA_PATH = "/mnt/pfs/loongflow/devmachine/1-04/evolux/output/mlebench/spooky-author-identification/prepared/public"
OUTPUT_DATA_PATH = "output/368bc6e8-482c-48b8-a870-040b0c3a264c/6/executor/output"

def load_data(validation_mode: bool = False) -> Tuple[X, y, X, Ids]:
    """
    Loads and prepares the Spooky Author Identification dataset.
    
    Args:
        validation_mode: If True, returns a subset of at most 200 rows for rapid testing.
            The subsetting is performed after full preparation to ensure pipeline consistency.
            
    Returns:
        Tuple[X, y, X, Ids]: (X_train, y_train, X_test, test_ids)
    """
    print(f"Starting data loading stage. Base path: {BASE_DATA_PATH}")

    # Step 0: Ensure output directory exists for any intermediate artifacts
    os.makedirs(OUTPUT_DATA_PATH, exist_ok=True)

    # Path definitions for the primary CSV files
    train_csv_path = os.path.join(BASE_DATA_PATH, "train.csv")
    test_csv_path = os.path.join(BASE_DATA_PATH, "test.csv")

    # Path verification: Fail fast if the expected files are missing.
    for path in [train_csv_path, test_csv_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing required dataset file: {path}")

    # Step 1: Load data from sources
    # Data size is small (~3.2 MB), using pandas with C engine is highly efficient.
    # Case and punctuation are preserved by default in standard CSV loading.
    try:
        df_train_full = pd.read_csv(train_csv_path, encoding='utf-8', engine='c')
        df_test_full = pd.read_csv(test_csv_path, encoding='utf-8', engine='c')
    except Exception as e:
        print(f"Error during CSV ingestion: {e}")
        raise

    # Step 2: Full Data Preparation
    # Ensure text columns are string type and preserve formatting.
    df_train_full['text'] = df_train_full['text'].astype(str)
    # Categorical encoding for 'author' for memory efficiency and downstream compatibility.
    df_train_full['author'] = df_train_full['author'].astype('category')
    
    df_test_full['text'] = df_test_full['text'].astype(str)
    df_test_full['id'] = df_test_full['id'].astype(str)

    # Step 3: Apply validation_mode subsetting
    if validation_mode:
        print("Validation mode enabled: subsetting data to 200 rows.")
        # Use head for representative subsetting while maintaining order.
        df_train = df_train_full.head(200).copy()
        df_test = df_test_full.head(200).copy()
    else:
        df_train = df_train_full
        df_test = df_test_full

    # Step 4: Structure data into required return format
    # X contains the features (text), y contains labels, Ids contains test identifiers.
    X_train = df_train[['text']]
    y_train = df_train['author']
    X_test = df_test[['text']]
    test_ids = df_test['id']

    # Final Integrity Checks: Ensure non-empty and row-aligned structures.
    if X_train.empty or y_train.empty or X_test.empty or test_ids.empty:
        raise ValueError("One or more returned data structures are empty.")
    
    if len(X_train) != len(y_train):
        raise ValueError(f"Training feature/target mismatch: {len(X_train)} vs {len(y_train)}")
    
    if len(X_test) != len(test_ids):
        raise ValueError(f"Test feature/ID mismatch: {len(X_test)} vs {len(test_ids)}")

    print(f"Successfully loaded data.")
    print(f"Training samples: {len(X_train)} | Test samples: {len(X_test)}")
    print(f"Target classes: {y_train.unique().tolist()}")

    return X_train, y_train, X_test, test_ids