import os
import pandas as pd
import numpy as np
from typing import Tuple, Dict, List, Union
from joblib import Parallel, delayed

# Task-adaptive type definitions
X = Dict[Union[int, str], pd.DataFrame]  # Dictionary mapping segment_id to seismic signal DataFrame
y = pd.Series                            # Target time_to_eruption indexed by segment_id
Ids = np.ndarray                         # Array of segment_ids for the test set

BASE_DATA_PATH = "/mnt/pfs/loongflow/devmachine/h20-09/evolux/output/mlebench/predict-volcanic-eruptions-ingv-oe/prepared/public"
OUTPUT_DATA_PATH = "output/bdc750a4-f0a3-4926-871d-f9675d7cf1ef/1/executor/output"

def _load_single_segment(segment_id: Union[int, str], folder: str) -> pd.DataFrame:
    """
    Helper function to load a single seismic segment CSV file.
    """
    file_path = os.path.join(BASE_DATA_PATH, folder, f"{segment_id}.csv")
    # Load as float32 to handle NaNs while keeping memory footprint reasonable
    return pd.read_csv(file_path, dtype='float32')

def load_data(validation_mode: bool = False) -> Tuple[X, y, X, Ids]:
    """
    Loads and prepares the datasets required for the volcanic eruption prediction task.
    Uses parallel processing to handle ~4,431 files efficiently.
    """
    print(f"Starting data loading (validation_mode={validation_mode})...")

    # 1. Load metadata
    train_metadata_path = os.path.join(BASE_DATA_PATH, "train.csv")
    sample_submission_path = os.path.join(BASE_DATA_PATH, "sample_submission.csv")
    
    train_df = pd.read_csv(train_metadata_path)
    test_metadata = pd.read_csv(sample_submission_path)

    # 2. Apply validation_mode subsetting
    if validation_mode:
        print("Subsetting data for validation mode...")
        train_df = train_df.head(200)
        test_metadata = test_metadata.head(200)

    train_ids = train_df['segment_id'].values
    y_train = train_df.set_index('segment_id')['time_to_eruption']
    test_ids = test_metadata['segment_id'].values

    # 3. Parallel loading of seismic signal files
    # Utilizing 36 available cores for maximum throughput
    n_jobs = 36
    
    print(f"Loading {len(train_ids)} training files in parallel...")
    train_signals = Parallel(n_jobs=n_jobs)(
        delayed(_load_single_segment)(sid, "train") for sid in train_ids
    )
    X_train = dict(zip(train_ids, train_signals))

    print(f"Loading {len(test_ids)} test files in parallel...")
    test_signals = Parallel(n_jobs=n_jobs)(
        delayed(_load_single_segment)(sid, "test") for sid in test_ids
    )
    X_test = dict(zip(test_ids, test_signals))

    # 4. Final verification of alignment
    assert len(X_train) == len(y_train), "Mismatch between X_train and y_train samples"
    assert len(X_test) == len(test_ids), "Mismatch between X_test and test_ids samples"
    
    print("Data loading completed successfully.")
    print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")
    
    return X_train, y_train, X_test, test_ids