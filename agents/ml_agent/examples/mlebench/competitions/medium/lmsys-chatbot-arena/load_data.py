import pandas as pd
import numpy as np
import os
from typing import Tuple

# Define the base data path as provided in the context
BASE_DATA_PATH = "/mnt/pfs/loongflow/devmachine/h20-01/evolux/output/mlebench/lmsys-chatbot-arena/prepared/public"
OUTPUT_DATA_PATH = "output/e051fb32-5ec2-4424-8b42-87dd7b28dacc/1/executor/output"

# Concrete type definitions for this task
X = pd.DataFrame  # Features: contains id, prompt, response_a, response_b
y = pd.Series     # Targets: single integer column [0, 1, 2]
Ids = pd.Series   # Identifiers: test set IDs for submission alignment

def load_data(validation_mode: bool = False) -> Tuple[X, y, X, Ids]:
    """
    Loads and prepares the datasets required for the LMSYS Chatbot Arena competition.
    Maps binary winner columns to a single integer label for multi-class classification.
    """
    # Step 0: Define paths and verify existence
    train_path = os.path.join(BASE_DATA_PATH, "train.csv")
    test_path = os.path.join(BASE_DATA_PATH, "test.csv")

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Training data not found at {train_path}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test data not found at {test_path}")

    # Step 1: Load data from sources
    # Use usecols for memory efficiency and lean execution
    train_cols = ['id', 'prompt', 'response_a', 'response_b', 'winner_model_a', 'winner_model_b', 'winner_tie']
    test_cols = ['id', 'prompt', 'response_a', 'response_b']

    print(f"Loading training data from {train_path}...")
    df_train = pd.read_csv(train_path, usecols=train_cols)
    
    print(f"Loading test data from {test_path}...")
    df_test = pd.read_csv(test_path, usecols=test_cols)

    # Step 2: Structure data into required return format
    # Map [winner_model_a, winner_model_b, winner_tie] to [0, 1, 2]
    # np.argmax works because columns are mutually exclusive and exhaustive
    target_cols = ['winner_model_a', 'winner_model_b', 'winner_tie']
    y_train_full = pd.Series(
        np.argmax(df_train[target_cols].values, axis=1), 
        name='label',
        index=df_train.index
    )
    
    # Feature matrices containing only relevant text and ID
    X_train_full = df_train[['id', 'prompt', 'response_a', 'response_b']].copy()
    X_test_full = df_test[['id', 'prompt', 'response_a', 'response_b']].copy()
    test_ids_full = df_test['id'].copy()

    # Step 3: Apply validation_mode subsetting if enabled
    if validation_mode:
        print("Validation mode: Subsetting data to 200 rows for quick execution.")
        X_train = X_train_full.head(200).reset_index(drop=True)
        y_train = y_train_full.head(200).reset_index(drop=True)
        X_test = X_test_full.head(200).reset_index(drop=True)
        test_ids = test_ids_full.head(200).reset_index(drop=True)
    else:
        X_train = X_train_full
        y_train = y_train_full
        X_test = X_test_full
        test_ids = test_ids_full

    # Step 4: Final verification and returns
    print(f"Successfully loaded data.")
    print(f"  Training samples: {len(X_train)}")
    print(f"  Test samples:     {len(X_test)}")
    
    # Row alignment checks
    if len(X_train) != len(y_train):
        raise ValueError(f"Alignment Error: X_train ({len(X_train)}) and y_train ({len(y_train)}) size mismatch.")
    if len(X_test) != len(test_ids):
        raise ValueError(f"Alignment Error: X_test ({len(X_test)}) and test_ids ({len(test_ids)}) size mismatch.")

    return X_train, y_train, X_test, test_ids