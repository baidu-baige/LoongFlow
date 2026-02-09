from typing import Tuple, Any
import pandas as pd
import numpy as np
import os

BASE_DATA_PATH = "/mnt/pfs/loongflow/devmachine/h20-04/evolux/output/mlebench/tabular-playground-series-dec-2021/prepared/public"
OUTPUT_DATA_PATH = "output/477e9955-ebee-46f4-96b0-878df6f022f5/1/executor/output"

# Task-adaptive type definitions
X = pd.DataFrame      # Feature matrix type
y = pd.Series         # Target vector type
Ids = pd.Series       # Identifier type for output alignment

def load_data(validation_mode: bool = False) -> Tuple[X, y, X, Ids]:
    """
    Loads and prepares the datasets required for the Forest Cover Type classification task.
    Implements memory optimization via downcasting and removes problematic samples.
    """
    train_path = os.path.join(BASE_DATA_PATH, "train.csv")
    test_path = os.path.join(BASE_DATA_PATH, "test.csv")

    # Step 0: Verify paths
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError(f"Required CSV files not found in {BASE_DATA_PATH}")

    # Step 1: Load data from sources
    print("Loading raw CSV files...")
    # Loading the full dataset first to ensure preparation is always complete
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    # Step 2: Structure and Prepare data
    print("Removing sample with Cover_Type == 5 (insufficient for CV)...")
    # Evidence: Class 5 has only 1 sample in the training set
    train = train[train['Cover_Type'] != 5].reset_index(drop=True)

    print("Executing memory optimization and type downcasting...")
    def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
        """Optimizes DataFrame memory footprint based on task specifications."""
        for col in df.columns:
            # Binary and target columns to int8
            if 'Wilderness_Area' in col or 'Soil_Type' in col or col == 'Cover_Type':
                df[col] = df[col].astype(np.int8)
            # Numeric integers to int32
            elif df[col].dtype == np.int64:
                df[col] = df[col].astype(np.int32)
            # Floating point to float32
            elif df[col].dtype == np.float64:
                df[col] = df[col].astype(np.float32)
        return df

    train = optimize_dtypes(train)
    test = optimize_dtypes(test)

    # Place prepared data into named subdirectories for pipeline audit
    prepared_dir = os.path.join(OUTPUT_DATA_PATH, "prepared_data")
    os.makedirs(prepared_dir, exist_ok=True)
    # We save a copy of the fully prepared (cleaned/downcast) data
    # Use parquet for efficiency if strictly for internal use, but CSV for general compatibility
    # Here we proceed with returning the objects to the next stage.

    # Step 3: Apply validation_mode subsetting if enabled
    if validation_mode:
        print("Validation mode active: subsetting to 200 representative rows.")
        # Taking head(200) ensures reproducibility and structure consistency
        train_subset = train.head(200)
        test_subset = test.head(200)
    else:
        train_subset = train
        test_subset = test

    # Define features, targets, and IDs
    y_train = train_subset['Cover_Type']
    X_train = train_subset.drop(columns=['Id', 'Cover_Type'])
    
    test_ids = test_subset['Id']
    X_test = test_subset.drop(columns=['Id'])

    # Step 4: Return aligned datasets
    print(f"Data loading complete.")
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}, test_ids shape: {test_ids.shape}")
    
    return X_train, y_train, X_test, test_ids