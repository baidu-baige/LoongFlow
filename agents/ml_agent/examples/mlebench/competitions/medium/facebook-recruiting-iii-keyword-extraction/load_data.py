import pandas as pd
import os
import pickle
import gc
from typing import Tuple, Any

BASE_DATA_PATH = "/mnt/pfs/loongflow/devmachine/h20-03/evolux/output/mlebench/facebook-recruiting-iii-keyword-extraction/prepared/public"
OUTPUT_DATA_PATH = "output/90689814-166b-4e4f-a971-355572d18239/1/executor/output"

# Define concrete types for this task
X = pd.DataFrame
y = pd.Series
Ids = pd.Series

def load_data(validation_mode: bool = False) -> Tuple[X, y, X, Ids]:
    """
    Loads and prepares the datasets for the Facebook Recruiting III keyword extraction task.
    Returns DataFrames and identifiers, attaching a duplicate lookup map to X_test.
    """
    train_csv = os.path.join(BASE_DATA_PATH, "train.csv")
    test_csv = os.path.join(BASE_DATA_PATH, "test.csv")
    
    # Step 0: Ensure data readiness - Prepare full duplicate map (Independent of validation_mode)
    prep_dir = os.path.join(BASE_DATA_PATH, "leakage_prepared")
    os.makedirs(prep_dir, exist_ok=True)
    map_path = os.path.join(prep_dir, "duplicate_map.pkl")
    
    if not os.path.exists(map_path):
        print("Preparing duplicate map from full training data...")
        # Load only necessary columns with efficient types
        full_train = pd.read_csv(train_csv, usecols=['Title', 'Body', 'Tags'], low_memory=True)
        
        # Consistent handling of missing values
        full_train['Title'] = full_train['Title'].fillna('').astype('string')
        full_train['Body'] = full_train['Body'].fillna('').astype('string')
        full_train['Tags'] = full_train['Tags'].fillna('').astype('string')
        
        # Deduplicate to create content -> tags mapping
        df_unique = full_train.drop_duplicates(subset=['Title', 'Body'])
        
        # Create dictionary: (Title, Body) -> Tags
        dup_map = dict(zip(zip(df_unique.Title, df_unique.Body), df_unique.Tags))
        
        with open(map_path, 'wb') as f:
            pickle.dump(dup_map, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        print(f"Duplicate map created with {len(dup_map)} unique entries.")
        del full_train, df_unique
        gc.collect()
    else:
        # We don't necessarily need to load it here if not validation_mode, 
        # but the spec implies returning it with the data.
        with open(map_path, 'rb') as f:
            dup_map = pickle.load(f)

    # Step 1: Load data from sources
    if validation_mode:
        print("Validation mode: Loading 200 rows.")
        train_df = pd.read_csv(train_csv, nrows=200, low_memory=True)
        test_df = pd.read_csv(test_csv, nrows=200, low_memory=True)
    else:
        print("Loading full datasets...")
        train_df = pd.read_csv(train_csv, low_memory=True)
        test_df = pd.read_csv(test_csv, low_memory=True)
        
        # Deduplicate training data
        train_df = train_df.drop_duplicates(subset=['Title', 'Body'])
        print(f"Training data deduplicated to {len(train_df)} rows.")

    # Apply string optimizations
    for df in [train_df, test_df]:
        df['Title'] = df['Title'].fillna('').astype('string')
        df['Body'] = df['Body'].fillna('').astype('string')
    if 'Tags' in train_df.columns:
        train_df['Tags'] = train_df['Tags'].fillna('').astype('string')

    # Step 2: Structure data into required return format
    X_train = train_df[['Title', 'Body']].copy()
    y_train = train_df['Tags'].copy()
    
    X_test = test_df[['Title', 'Body']].copy()
    test_ids = test_df['Id'].copy()

    # If duplicate_map was not loaded in validation_mode branch, load it now
    if 'dup_map' not in locals():
        with open(map_path, 'rb') as f:
            dup_map = pickle.load(f)

    # Attach duplicate_map to X_test metadata for downstream ensemble use
    # Using .attrs is the standard way to attach metadata to pandas objects
    X_test.attrs['duplicate_map'] = dup_map

    # Step 3: Verify alignment
    assert len(X_train) == len(y_train), f"X_train ({len(X_train)}) and y_train ({len(y_train)}) mismatch."
    assert len(X_test) == len(test_ids), f"X_test ({len(X_test)}) and test_ids ({len(test_ids)}) mismatch."

    print(f"Successfully loaded data. Train: {len(X_train)}, Test: {len(X_test)}")
    
    # Step 4: Return X_train, y_train, X_test, test_ids
    return X_train, y_train, X_test, test_ids