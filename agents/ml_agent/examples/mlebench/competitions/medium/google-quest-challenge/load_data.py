import os
import pandas as pd
import unicodedata
from typing import Tuple

# Task-adaptive type definitions
X = pd.DataFrame      # Feature matrix containing text and categorical metadata
y = pd.DataFrame      # Target vector containing 30 continuous labels [0,1]
Ids = pd.Series       # Identifier series (qa_id) for alignment and submission

BASE_DATA_PATH = "/mnt/pfs/loongflow/devmachine/2-05/evolux/output/mlebench/google-quest-challenge/prepared/public"
OUTPUT_DATA_PATH = "output/aaa741b3-cb02-44fc-a666-dd434e563444/8/executor/output"

def load_data(validation_mode: bool = False) -> Tuple[X, y, X, Ids]:
    """
    Loads and prepares the Google QUEST Q&A Labeling datasets.
    Performs full data ingestion, NFKC normalization, and persists the processed state.

    Args:
        validation_mode: If True, returns a representative subset of up to 200 rows.
        
    Returns:
        Tuple[X_train, y_train, X_test, test_ids]:
            - X_train: Training features (text columns + metadata)
            - y_train: 30 target labels [0, 1]
            - X_test: Test features
            - test_ids: qa_id for alignment in sub-stages
    """
    print(f"Execution Stage: load_data | Validation Mode: {validation_mode}")

    # Step 0: Define and verify paths
    train_csv = os.path.join(BASE_DATA_PATH, "train.csv")
    test_csv = os.path.join(BASE_DATA_PATH, "test.csv")
    sample_sub_csv = os.path.join(BASE_DATA_PATH, "sample_submission.csv")

    for path in [train_csv, test_csv, sample_sub_csv]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Critical data file missing at path: {path}")

    # Step 1: Load raw data
    # Using 'c' engine for optimized performance given the hardware context
    print("Loading CSV datasets...")
    train_df = pd.read_csv(train_csv, engine='c', encoding='utf-8')
    test_df = pd.read_csv(test_csv, engine='c', encoding='utf-8')
    sample_sub = pd.read_csv(sample_sub_csv, engine='c', encoding='utf-8')

    # Identify target columns from sample submission
    target_cols = [col for col in sample_sub.columns if col != 'qa_id']
    
    # Step 2: Full Data Preparation
    # NFKC normalization handles varied unicode representations, ensuring dual-tokenization consistency
    def normalize_text(text):
        if pd.isna(text):
            return ""
        return unicodedata.normalize('NFKC', str(text)).strip()

    text_cols = ['question_title', 'question_body', 'answer']
    print(f"Normalizing text features (NFKC): {text_cols}")
    
    for col in text_cols:
        if col in train_df.columns:
            train_df[col] = train_df[col].apply(normalize_text)
        if col in test_df.columns:
            test_df[col] = test_df[col].apply(normalize_text)

    # Persist prepared data to output for downstream stage efficiency
    prepared_dir = os.path.join(OUTPUT_DATA_PATH, "prepared_data")
    os.makedirs(prepared_dir, exist_ok=True)
    
    train_prepared_path = os.path.join(prepared_dir, "train_prepared.parquet")
    test_prepared_path = os.path.join(prepared_dir, "test_prepared.parquet")
    
    # Parquet utilizes efficient columnar storage
    train_df.to_parquet(train_prepared_path, index=False)
    test_df.to_parquet(test_prepared_path, index=False)
    print(f"Successfully persisted full prepared data to {prepared_dir}")

    # Step 3: Structure data into return format
    # Split training into features (X) and targets (y)
    y_train_full = train_df[target_cols].copy()
    X_train_full = train_df.drop(columns=target_cols)
    
    # Test set features and IDs
    X_test_full = test_df.copy()
    test_ids_full = test_df['qa_id'].copy()

    # Step 4: Apply subsetting for validation mode
    if validation_mode:
        subset_size = min(200, len(X_train_full), len(X_test_full))
        print(f"Validation mode enabled: Subsetting to top {subset_size} rows.")
        X_train = X_train_full.head(subset_size).reset_index(drop=True)
        y_train = y_train_full.head(subset_size).reset_index(drop=True)
        X_test = X_test_full.head(subset_size).reset_index(drop=True)
        test_ids = test_ids_full.head(subset_size).reset_index(drop=True)
    else:
        X_train = X_train_full
        y_train = y_train_full
        X_test = X_test_full
        test_ids = test_ids_full

    # Final verification of alignment
    if len(X_train) != len(y_train) or len(X_test) != len(test_ids):
        raise ValueError("Dimensional misalignment detected in prepared datasets.")

    print(f"Data loading complete.")
    print(f"Train samples: {X_train.shape[0]} | Test samples: {X_test.shape[0]}")
    print(f"Target count: {y_train.shape[1]}")
    
    return X_train, y_train, X_test, test_ids