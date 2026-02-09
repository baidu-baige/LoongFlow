import os
import cudf
from typing import Tuple

BASE_DATA_PATH = "/mnt/pfs/loongflow/devmachine/2-05/evolux/output/mlebench/learning-agency-lab-automated-essay-scoring-2/prepared/public"
OUTPUT_DATA_PATH = "output/60558907-7467-4c6f-a8d6-3b62cb9514db/1/executor/output"

# Task-adaptive type definitions using cudf for GPU acceleration
X = cudf.DataFrame
y = cudf.Series
Ids = cudf.Series

def load_data(validation_mode: bool = False) -> Tuple[X, y, X, Ids]:
    """
    Loads and prepares the Automated Essay Scoring dataset.
    Uses GPU-accelerated processing via cudf to efficiently handle text normalization.
    """
    print(f"Stage 1: load_data starting. Validation Mode: {validation_mode}")

    # Paths for raw data and prepared cache
    train_csv_path = os.path.join(BASE_DATA_PATH, "train.csv")
    test_csv_path = os.path.join(BASE_DATA_PATH, "test.csv")
    prepared_dir = os.path.join(OUTPUT_DATA_PATH, "prepared_data")
    os.makedirs(prepared_dir, exist_ok=True)
    
    train_parquet_path = os.path.join(prepared_dir, "train_cleaned.parquet")
    test_parquet_path = os.path.join(prepared_dir, "test_cleaned.parquet")

    # Step 0 & 1: Ensure data readiness and Load
    # We check if cleaned data already exists in Parquet format for I/O efficiency
    if os.path.exists(train_parquet_path) and os.path.exists(test_parquet_path):
        print("Loading full prepared data from Parquet cache...")
        train_full = cudf.read_parquet(train_parquet_path)
        test_full = cudf.read_parquet(test_parquet_path)
    else:
        print("Prepared data not found. Processing raw CSV files...")
        if not os.path.exists(train_csv_path) or not os.path.exists(test_csv_path):
            raise FileNotFoundError(f"Raw data files not found in {BASE_DATA_PATH}. "
                                    f"Expected train.csv and test.csv.")

        # Load raw data into GPU memory
        train_full = cudf.read_csv(train_csv_path)
        test_full = cudf.read_csv(test_csv_path)

        # Step 2: Basic text normalization as per Technical Specification
        # Objectives: Normalize whitespace (multiple spaces/newlines -> single equivalents)
        print("Normalizing essay text on GPU...")
        for df in [train_full, test_full]:
            # Replace multiple spaces with a single space
            df['full_text'] = df['full_text'].str.replace(r' +', ' ', regex=True)
            # Replace multiple newlines or carriage returns with a single newline
            df['full_text'] = df['full_text'].str.replace(r'[\r\n]+', '\n', regex=True)
            # Strip leading and trailing whitespace
            df['full_text'] = df['full_text'].str.strip()

        # Save prepared data to disk to avoid redundant computation in future runs
        # Data preparation is always full, independent of validation_mode
        train_full.to_parquet(train_parquet_path)
        test_full.to_parquet(test_parquet_path)
        print(f"Prepared full datasets cached to {prepared_dir}")

    # Step 3: Apply validation_mode subsetting if enabled
    if validation_mode:
        print("Validation mode enabled: Returning representative subset (200 rows).")
        train_df = train_full.head(200)
        test_df = test_full.head(200)
    else:
        train_df = train_full
        test_df = test_full

    # Step 4: Structure data into required return format
    # X_train: Feature matrix (text)
    # y_train: Target vector (score)
    # X_test: Feature matrix for inference
    # test_ids: Identifiers for final submission alignment
    X_train = train_df[['full_text']]
    y_train = train_df['score']
    X_test = test_df[['full_text']]
    test_ids = test_df['essay_id']

    # Final sanity checks
    if len(X_train) == 0 or len(y_train) == 0 or len(X_test) == 0 or len(test_ids) == 0:
        raise ValueError("One or more return datasets are empty. Check data source integrity.")
    
    if len(X_train) != len(y_train):
        raise ValueError(f"Train feature/target misalignment: X={len(X_train)}, y={len(y_train)}")
        
    if len(X_test) != len(test_ids):
        raise ValueError(f"Test feature/id misalignment: X={len(X_test)}, ids={len(test_ids)}")

    print(f"Data loading complete. Train samples: {len(X_train)}, Test samples: {len(X_test)}")
    return X_train, y_train, X_test, test_ids