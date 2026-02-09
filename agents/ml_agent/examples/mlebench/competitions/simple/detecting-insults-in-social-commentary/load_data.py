import pandas as pd
import numpy as np
import os
from typing import Tuple, Any

# Task-adaptive type definitions
# X: Features (DataFrame containing Comment and Date)
# y: Target (Series of binary labels)
# Ids: Identifiers for test rows (Series of integers)
X = pd.DataFrame
y = pd.Series
Ids = pd.Series

BASE_DATA_PATH = "/mnt/pfs/loongflow/devmachine/2-05/evolux/output/mlebench/detecting-insults-in-social-commentary/prepared/public"


def clean_comment(comment: Any) -> str:
    """
    Unescapes unicode sequences in the text as per the technical specification.
    Example: "\\u00a0" -> "\xa0"
    """
    if not isinstance(comment, str):
        return ""
    try:
        # Technical Specification requirement: bytes(comment, "utf-8").decode("unicode_escape")
        # This handles the literal unicode-escaped characters present in the raw CSV.
        return bytes(comment, "utf-8").decode("unicode_escape")
    except Exception:
        # Fallback to original string if decoding fails to avoid data loss
        return str(comment)


def load_data(validation_mode: bool = False) -> Tuple[X, y, X, Ids]:
    """
    Loads and prepares the datasets required for the insult detection task.

    Args:
        validation_mode: Controls the data loading behavior.
            - False (default): Load the complete dataset.
            - True: Return a small subset of data (<= 200 rows) for quick validation.

    Returns:
        Tuple[X, y, X, Ids]: A tuple containing four elements:
        - X_train (X): Training features (Comment, Date).
        - y_train (y): Training targets (Insult).
        - X_test (X): Test features (Comment, Date).
        - test_ids (Ids): Identifiers for generating the submission.
    """
    print("Stage 1: Starting data loading and preparation...")

    # Define paths
    train_path = os.path.join(BASE_DATA_PATH, "train.csv")
    test_path = os.path.join(BASE_DATA_PATH, "test.csv")

    # Step 0: Verify data readiness
    # Sampling verify existence to ensure paths align with actual directory structure
    for path in [train_path, test_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required raw data file not found: {path}")

    # Step 1: Load raw data from source CSV files
    # We use low_memory=False to ensure type consistency for larger datasets
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    print(f"Initial load complete. Train: {len(df_train)} rows. Test: {len(df_test)} rows.")

    # Step 2: Data preparation (Full preparation independent of validation_mode)
    def prepare_df(df: pd.DataFrame) -> pd.DataFrame:
        # Create a copy to prevent SettingWithCopy warnings if subsetted later
        df = df.copy()

        # Resolve 'Comment' unicode escaping
        if 'Comment' in df.columns:
            # Apply cleaning function to each row
            df['Comment'] = df['Comment'].apply(clean_comment)

        # Parse 'Date' in "YYYYMMDDhhmmssZ" format
        if 'Date' in df.columns:
            # errors='coerce' turns invalid or blank dates into NaT (Not a Time)
            df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d%H%M%SZ', errors='coerce')

        return df

    print("Cleaning text and parsing temporal attributes...")
    df_train = prepare_df(df_train)
    df_test = prepare_df(df_test)

    # Step 3: Structure data into required return format
    if 'Insult' not in df_train.columns:
        raise ValueError("Critical error: Target column 'Insult' missing from training data.")

    y_train = df_train['Insult'].astype(np.int64)
    X_train = df_train.drop(columns=['Insult'])
    X_test = df_test

    # test_ids: Use the row index as identifiers for test samples
    test_ids = pd.Series(df_test.index, name='id')

    # Step 4: Apply validation_mode subsetting if enabled
    if validation_mode:
        print("Validation Mode: Subsetting datasets to 200 rows for rapid execution.")
        X_train = X_train.head(200)
        y_train = y_train.head(200)
        X_test = X_test.head(200)
        test_ids = test_ids.head(200)

    # Step 5: Final integrity checks
    if X_train.empty or y_train.empty or X_test.empty or test_ids.empty:
        raise RuntimeError("Load failed: One or more output variables are empty.")

    if len(X_train) != len(y_train):
        raise RuntimeError(f"Alignment Error: X_train ({len(X_train)}) and y_train ({len(y_train)}) mismatch.")

    if len(X_test) != len(test_ids):
        raise RuntimeError(f"Alignment Error: X_test ({len(X_test)}) and test_ids ({len(test_ids)}) mismatch.")

    print(f"Data loading complete. Train features shape: {X_train.shape}. Test features shape: {X_test.shape}.")

    return X_train, y_train, X_test, test_ids