import pandas as pd
import numpy as np
import os
from typing import Tuple

BASE_DATA_PATH = "/mnt/pfs/loongflow/devmachine/h20-04/evolux/output/mlebench/freesound-audio-tagging-2019/prepared/public"
OUTPUT_DATA_PATH = "output/41d6445d-6924-4bdd-a657-0e220176048a/2/executor/output"

# Concrete types for this task
X = pd.DataFrame  # Contains metadata including file paths and curated flags
y = pd.DataFrame  # Multi-hot encoded 80-category labels
Ids = pd.Series   # Filenames for submission alignment

def load_data(validation_mode: bool = False) -> Tuple[X, y, X, Ids]:
    """
    Loads and prepares the Freesound Audio Tagging 2019 datasets.
    
    Args:
        validation_mode: If True, returns a small subset (max 200 rows) for quick testing.
        
    Returns:
        X_train: Metadata for training files (fname, path, labels, is_curated).
        y_train: Multi-hot target matrix (80 columns).
        X_test: Metadata for test files (fname, path).
        test_ids: Original filenames for the test set.
    """
    # 1. Define base paths for files and directories
    train_curated_csv = os.path.join(BASE_DATA_PATH, "train_curated.csv")
    train_noisy_csv = os.path.join(BASE_DATA_PATH, "train_noisy.csv")
    sample_sub_csv = os.path.join(BASE_DATA_PATH, "sample_submission.csv")
    
    train_curated_dir = os.path.join(BASE_DATA_PATH, "train_curated")
    train_noisy_dir = os.path.join(BASE_DATA_PATH, "train_noisy")
    test_dir = os.path.join(BASE_DATA_PATH, "test")

    # 2. Get the standardized 80 labels from sample_submission.csv
    print("Standardizing label names and order from sample_submission.csv...")
    sample_sub = pd.read_csv(sample_sub_csv)
    label_columns = sample_sub.columns[1:].tolist()
    label_to_idx = {label: i for i, label in enumerate(label_columns)}
    
    # 3. Identify corrupted files to exclude (from competition notes)
    # Wrong labels: f76181c4.wav, 77b925c2.wav, 6a1f682a.wav, c7db12aa.wav, 7752cc8a.wav
    # No signal: 1d44b0bd.wav
    corrupted_files = {
        'f76181c4.wav', '77b925c2.wav', '6a1f682a.wav', 
        'c7db12aa.wav', '7752cc8a.wav', '1d44b0bd.wav'
    }

    def process_split(csv_path: str, audio_dir: str, is_curated: bool) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Metadata file not found: {csv_path}")
            
        df = pd.read_csv(csv_path)
        
        # Remove corrupted files identified in data description
        initial_count = len(df)
        df = df[~df['fname'].isin(corrupted_files)].reset_index(drop=True)
        removed_count = initial_count - len(df)
        if removed_count > 0:
            print(f"Removed {removed_count} corrupted files from subset.")
        
        # Construct absolute paths for efficient loading
        df['path'] = df['fname'].apply(lambda x: os.path.join(audio_dir, x))
        
        # Verify a sample file existence to catch directory structure issues early
        if not df.empty and not os.path.exists(df['path'].iloc[0]):
            raise FileNotFoundError(f"Audio path verification failed: {df['path'].iloc[0]}")
        
        # Multi-hot encode the labels (80 columns)
        # Using numpy for efficiency during encoding
        multi_hot = np.zeros((len(df), len(label_columns)), dtype=np.float32)
        for i, labels_str in enumerate(df['labels']):
            labels_list = labels_str.split(',')
            for l in labels_list:
                if l in label_to_idx:
                    multi_hot[i, label_to_idx[l]] = 1.0
        
        y_df = pd.DataFrame(multi_hot, columns=label_columns)
        
        # Track whether the sample comes from the curated or noisy subset
        df['is_curated'] = is_curated
        
        return df, y_df

    # 4. Load training subsets
    print("Processing curated subset...")
    X_cur, y_cur = process_split(train_curated_csv, train_curated_dir, is_curated=True)
    
    print("Processing noisy subset...")
    X_noi, y_noi = process_split(train_noisy_csv, train_noisy_dir, is_curated=False)

    # 5. Concatenate for final training set
    X_train = pd.concat([X_cur, X_noi], axis=0).reset_index(drop=True)
    y_train = pd.concat([y_cur, y_noisy if 'y_noisy' in locals() else y_noi], axis=0).reset_index(drop=True)

    # 6. Prepare test metadata
    print("Preparing test metadata...")
    X_test = sample_sub[['fname']].copy()
    X_test['path'] = X_test['fname'].apply(lambda x: os.path.join(test_dir, x))
    test_ids = X_test['fname']

    # 7. Handle validation mode subsetting
    if validation_mode:
        print("Validation mode: Subsetting data to ≤ 200 samples.")
        # Ensure representation by picking from both curated and noisy subsets
        cur_idx = X_train[X_train['is_curated']].index[:100].tolist()
        noi_idx = X_train[~X_train['is_curated']].index[:100].tolist()
        val_indices = cur_idx + noi_idx
        
        X_train = X_train.loc[val_indices].reset_index(drop=True)
        y_train = y_train.loc[val_indices].reset_index(drop=True)
        
        # Subset test data
        X_test = X_test.head(200).reset_index(drop=True)
        test_ids = test_ids.head(200).reset_index(drop=True)

    print(f"Data loading complete. Train: {len(X_train)} samples, Test: {len(X_test)} samples.")
    return X_train, y_train, X_test, test_ids