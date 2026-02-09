import os
import zipfile
import cudf
from typing import Tuple

# Task-adaptive type definitions using RAPIDS cuDF for high-performance GPU data processing.
# cuDF is chosen to handle the large scale (~10M rows) of the Russian text normalization dataset
# efficiently within the available 140GB H20-3e GPU memory and 440GB system RAM.
X = cudf.DataFrame
y = cudf.Series
Ids = cudf.Series

BASE_DATA_PATH = "/mnt/pfs/loongflow/devmachine/h20-12/evolux/output/mlebench/text-normalization-challenge-russian-language/prepared/public"
OUTPUT_DATA_PATH = "output/ecfe1a48-59fb-4170-a38b-6ffb4a298ec0/10/executor/output"
PREPARED_DIR = os.path.join(BASE_DATA_PATH, "prepared_optimized")

def load_data(validation_mode: bool = False) -> Tuple[X, y, X, Ids]:
    """
    Loads and prepares the datasets required for the Russian Text Normalization task.
    Uses GPU acceleration to minimize I/O and processing bottlenecks.
    """
    # Step 0: Ensure data readiness - prepare full data if missing
    # Data preparation must be complete regardless of validation_mode
    os.makedirs(PREPARED_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DATA_PATH, exist_ok=True)

    train_zip = os.path.join(BASE_DATA_PATH, "ru_train.csv.zip")
    test_zip = os.path.join(BASE_DATA_PATH, "ru_test_2.csv.zip")
    
    # Filenames based on dataset structure
    train_csv = os.path.join(PREPARED_DIR, "ru_train.csv")
    test_csv = os.path.join(PREPARED_DIR, "ru_test_2.csv")

    # Extract training data if not already present
    if not os.path.exists(train_csv):
        print(f"Extracting {train_zip} to {PREPARED_DIR}...")
        if not os.path.exists(train_zip):
            raise FileNotFoundError(f"Source zip not found: {train_zip}")
        with zipfile.ZipFile(train_zip, 'r') as z:
            z.extractall(PREPARED_DIR)
    
    # Extract test data if not already present
    if not os.path.exists(test_csv):
        print(f"Extracting {test_zip} to {PREPARED_DIR}...")
        if not os.path.exists(test_zip):
            raise FileNotFoundError(f"Source zip not found: {test_zip}")
        with zipfile.ZipFile(test_zip, 'r') as z:
            z.extractall(PREPARED_DIR)

    # Step 1: Load data from sources
    # Use compact dtypes for memory efficiency:
    # sentence_id (up to ~700k) -> int32
    # token_id (up to ~700) -> int16
    # class -> category (15 distinct classes)
    train_dtypes = {
        'sentence_id': 'int32',
        'token_id': 'int16',
        'class': 'category',
        'before': 'string',
        'after': 'string'
    }
    
    test_dtypes = {
        'sentence_id': 'int32',
        'token_id': 'int16',
        'before': 'string'
    }

    # Apply validation subset logic during the read stage for performance
    nrows = 200 if validation_mode else None

    print(f"Loading training data (nrows={nrows})...")
    df_train = cudf.read_csv(train_csv, dtype=train_dtypes, nrows=nrows)
    
    print(f"Loading test data (nrows={nrows})...")
    # Note: ru_test_2.csv matches the column structure needed for inference
    df_test = cudf.read_csv(test_csv, dtype=test_dtypes, nrows=nrows)

    # Step 2: Structure data into required return format
    # X_train includes 'class' to allow semiotic-aware modeling or filtering in downstream stages
    X_train = df_train[['sentence_id', 'token_id', 'before', 'class']]
    y_train = df_train['after']
    
    # X_test matches features available at inference time
    X_test = df_test[['sentence_id', 'token_id', 'before']]
    
    # Step 3: Create submission identifiers
    # The competition requires 'id' formatted as 'sentence_id_token_id' (e.g., "123_5")
    test_ids = df_test['sentence_id'].astype('str') + "_" + df_test['token_id'].astype('str')

    # Step 4: Verification and Cleanup
    if X_train.empty or y_train.empty or X_test.empty or test_ids.empty:
        raise ValueError("One or more loaded datasets are empty. Check data source integrity.")

    # Memory Cleanup: Free the full DataFrames as we have extracted the necessary components
    del df_train
    del df_test

    print(f"Data loading complete.")
    print(f" - X_train shape: {X_train.shape}")
    print(f" - y_train shape: {y_train.shape}")
    print(f" - X_test shape:  {X_test.shape}")
    print(f" - test_ids shape: {test_ids.shape}")

    return X_train, y_train, X_test, test_ids