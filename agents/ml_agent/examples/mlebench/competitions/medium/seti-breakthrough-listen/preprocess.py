import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from typing import Tuple

BASE_DATA_PATH = "/mnt/pfs/loongflow/devmachine/h20-09/evolux/output/mlebench/seti-breakthrough-listen/prepared/public"
OUTPUT_DATA_PATH = "output/a429c40e-fe12-455c-8b05-ca9d732aabeb/1/executor/output"

# Task-adaptive type definitions
X = np.ndarray  # Feature matrix of shape (N, 1, 1638, 256)
y = np.ndarray  # Target vector of shape (N,)

def load_and_stack(filepath: str) -> np.ndarray:
    """
    Loads a single cadence snippet, performs vertical stacking of the 6 scans,
    and ensures the format is model-ready with a channel dimension.
    
    Args:
        filepath (str): Path to the .npy file.
        
    Returns:
        np.ndarray: Stacked cadence of shape (1, 1638, 256).
    """
    # Load the cadence snippet: shape (6, 273, 256)
    # The data is stored as float16; cast to float32 for model training and consistency
    data = np.load(filepath).astype(np.float32)
    
    # Method: Vertical Stacking
    # Concatenate the 6 scans (axis 0) along the frequency dimension (axis 1)
    # np.vstack iterates over the first axis of the input array.
    # Resulting shape: (6 * 273, 256) = (1638, 256)
    stacked = np.vstack(data)
    
    # Add channel dimension for backbone compatibility: (1, 1638, 256)
    return stacked[np.newaxis, ...]

def preprocess(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame
) -> Tuple[X, y, X, y, X]:
    """
    Transforms raw data into model-ready format for a single fold/split.
    
    This implementation performs vertical stacking of the 6 cadence scans to create 
    a single image representing the 'on-off' pattern across A-B-A-C-A-D targets.
    Processing is parallelized across all available CPU cores for efficiency.
    """
    # Step 1: Process training data
    print(f"Preprocessing training set: {len(X_train)} samples...")
    X_train_processed = np.stack(Parallel(n_jobs=36)(
        delayed(load_and_stack)(fp) for fp in X_train['filepath']
    ))
    y_train_processed = y_train.values.astype(np.int64)

    # Step 2: Process validation data
    print(f"Preprocessing validation set: {len(X_val)} samples...")
    X_val_processed = np.stack(Parallel(n_jobs=36)(
        delayed(load_and_stack)(fp) for fp in X_val['filepath']
    ))
    y_val_processed = y_val.values.astype(np.int64)

    # Step 3: Process test data (completeness check: covers all X_test entries)
    print(f"Preprocessing test set: {len(X_test)} samples...")
    X_test_processed = np.stack(Parallel(n_jobs=36)(
        delayed(load_and_stack)(fp) for fp in X_test['filepath']
    ))

    # Step 4: Validate output format and quality
    # Ensure no NaN or Infinity values exist in the processed arrays
    if not np.isfinite(X_train_processed).all():
        raise ValueError("Detected non-finite values (NaN/Inf) in processed X_train.")
    if not np.isfinite(X_val_processed).all():
        raise ValueError("Detected non-finite values (NaN/Inf) in processed X_val.")
    if not np.isfinite(X_test_processed).all():
        raise ValueError("Detected non-finite values (NaN/Inf) in processed X_test.")

    # Consistency checks
    assert len(X_train_processed) == len(y_train_processed), "Train features and targets mismatch."
    assert len(X_val_processed) == len(y_val_processed), "Val features and targets mismatch."
    assert X_train_processed.shape[1:] == (1, 1638, 256), f"Unexpected shape: {X_train_processed.shape}"

    print(f"Preprocessing complete. Final shapes: "
          f"Train={X_train_processed.shape}, Val={X_val_processed.shape}, Test={X_test_processed.shape}")

    return (
        X_train_processed,
        y_train_processed,
        X_val_processed,
        y_val_processed,
        X_test_processed
    )