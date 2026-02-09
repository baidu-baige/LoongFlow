import pandas as pd
import os
from typing import Tuple

BASE_DATA_PATH = "/mnt/pfs/loongflow/devmachine/h20-04/evolux/output/mlebench/text-normalization-challenge-english-language/prepared/public"
OUTPUT_DATA_PATH = "output/9fef8e79-9e97-4657-be88-07dd4ac6f366/1/executor/output"

# Task-adaptive type definitions
X = pd.DataFrame
y = pd.Series
Ids = pd.Series

def load_data(validation_mode: bool = False) -> Tuple[X, y, X, Ids]:
    """
    Loads and prepares the datasets for the English Text Normalization Challenge.
    Maintaining sentence-token hierarchy by preserving IDs.
    """
    train_path = os.path.join(BASE_DATA_PATH, "en_train.csv.zip")
    test_path = os.path.join(BASE_DATA_PATH, "en_test_2.csv.zip")
    
    # Verify mandatory files exist
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Missing training data at: {train_path}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Missing test data at: {test_path}")

    # Explicit dtypes for memory efficiency and technical specification compliance
    # 'class' is omitted from test but provided in train for training context/multi-tasking
    train_dtypes = {
        'sentence_id': 'int32',
        'token_id': 'int16',
        'before': 'str',
        'class': 'category',
        'after': 'str'
    }
    test_dtypes = {
        'sentence_id': 'int32',
        'token_id': 'int16',
        'before': 'str'
    }

    # Load data with keep_default_na=False to prevent interpretation of tokens like "NA" or "null" as NaN
    print(f"Loading datasets (validation_mode={validation_mode})...")
    
    try:
        # Load train set
        train_df = pd.read_csv(
            train_path, 
            compression='zip', 
            dtype=train_dtypes, 
            keep_default_na=False,
            nrows=200 if validation_mode else None
        )
        
        # Load test set
        test_df = pd.read_csv(
            test_path, 
            compression='zip', 
            dtype=test_dtypes, 
            keep_default_na=False, 
            nrows=200 if validation_mode else None
        )
    except Exception as e:
        print(f"Error during file loading: {e}")
        raise

    # Structure Training data
    # Preserve hierarchy via sentence_id and token_id
    X_train = train_df[['sentence_id', 'token_id', 'before', 'class']]
    y_train = train_df['after']

    # Structure Test data
    X_test = test_df[['sentence_id', 'token_id', 'before']]
    
    # Generate submission format IDs: "{sentence_id}_{token_id}"
    # This aligns with the requirement for test_ids to map predictions to output format.
    test_ids = test_df['sentence_id'].astype(str) + "_" + test_df['token_id'].astype(str)
    test_ids.name = 'id'

    # Performance logging
    print(f"Completed loading.")
    print(f"Train samples: {len(X_train)} | Test samples: {len(X_test)}")
    print(f"Memory Usage (Train): {X_train.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

    return X_train, y_train, X_test, test_ids