import pandas as pd
import os
from typing import Tuple

# Task-adaptive type definitions
X = pd.DataFrame      # Feature matrix: RNA molecules with sequence/structure attributes
y = pd.DataFrame      # Target vector: Nested lists of degradation/reactivity values
Ids = pd.Series       # Identifier: Sample IDs for alignment

BASE_DATA_PATH = "/mnt/pfs/loongflow/devmachine/h20-11/evolux/output/mlebench/stanford-covid-vaccine/prepared/public"
OUTPUT_DATA_PATH = "output/cd1762a7-cbef-43b4-bbfc-bc919a0a7546/1/executor/output"

def load_data(validation_mode: bool = False) -> Tuple[X, y, X, Ids]:
    """
    Loads and prepares the mRNA degradation datasets.
    
    The function loads RNA sequences, their estimated secondary structures, and 
    associated experimental degradation rates (targets) from JSON files.
    """
    
    # Define paths
    train_path = os.path.join(BASE_DATA_PATH, "train.json")
    test_path = os.path.join(BASE_DATA_PATH, "test.json")
    
    # Step 0: Ensure data readiness
    # Check for existence of essential files
    for path in [train_path, test_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required data file not found at: {path}")

    print(f"Loading raw data from {BASE_DATA_PATH}...")
    
    # Step 1: Load data from sources
    # Using pandas read_json with lines=True as the files are JSONL format
    df_train_raw = pd.read_json(train_path, lines=True)
    df_test_raw = pd.read_json(test_path, lines=True)
    
    # Step 2: Structure data into required return format
    
    # Filter training data based on SN_filter as per technical specification
    # SN_filter indicates if the sample passed quality filters (signal-to-noise, etc.)
    print(f"Initial training samples: {len(df_train_raw)}")
    df_train = df_train_raw[df_train_raw['SN_filter'] == 1].reset_index(drop=True)
    print(f"Training samples after SN_filter == 1: {len(df_train)}")
    
    # Define feature and target columns
    # Features common to both train and test
    feature_cols = [
        'id', 
        'sequence', 
        'structure', 
        'predicted_loop_type', 
        'seq_length', 
        'seq_scored'
    ]
    
    # Ground truth targets (lists of floats for the first seq_scored bases)
    target_cols = [
        'reactivity', 
        'deg_Mg_pH10', 
        'deg_pH10', 
        'deg_Mg_50C', 
        'deg_50C'
    ]
    
    # Step 3: Apply validation_mode subsetting if enabled
    if validation_mode:
        print("Validation mode: Subsetting data to at most 200 rows.")
        df_train = df_train.head(200)
        df_test_raw = df_test_raw.head(200)
        
    # Extract feature matrices and labels
    # We use .copy() to ensure we return independent objects and avoid SettingWithCopy warnings downstream
    X_train = df_train[feature_cols].copy()
    y_train = df_train[target_cols].copy()
    X_test = df_test_raw[feature_cols].copy()
    test_ids = df_test_raw['id'].copy()
    
    # Verify non-empty and alignment
    if X_train.empty or y_train.empty or X_test.empty or test_ids.empty:
        raise ValueError("Data loading resulted in empty datasets. Check source files or filtering logic.")
        
    if len(X_train) != len(y_train):
        raise ValueError(f"Train features and targets mismatch: {len(X_train)} vs {len(y_train)}")
        
    if len(X_test) != len(test_ids):
        raise ValueError(f"Test features and IDs mismatch: {len(X_test)} vs {len(test_ids)}")

    print(f"Successfully loaded {len(X_train)} training samples and {len(X_test)} test samples.")
    print(f"Features: {X_train.columns.tolist()}")
    print(f"Targets: {y_train.columns.tolist()}")

    # Step 4: Return X_train, y_train, X_test, test_ids
    return X_train, y_train, X_test, test_ids