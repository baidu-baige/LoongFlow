import os
import pandas as pd
from glob import glob
from typing import Tuple

# Task-adaptive type definitions
X = pd.DataFrame      # Feature matrix type (contains file_path and metadata)
y = pd.Series         # Target vector type (labels)
Ids = pd.Series       # Identifier type for output alignment (filenames)

BASE_DATA_PATH = "/mnt/pfs/loongflow/devmachine/2-05/evolux/output/mlebench/tensorflow-speech-recognition-challenge/prepared/public"
OUTPUT_DATA_PATH = "output/fe927f25-451a-41da-a547-cdb392b784d8/1/executor/output"

def load_data(validation_mode: bool = False) -> Tuple[X, y, X, Ids]:
    """
    Loads and prepares the datasets required for the speech recognition task.
    Collects 30 word classes and silence samples, extracting speaker IDs for grouping.
    """
    print(f"Starting data loading process. Validation mode: {validation_mode}")

    # Step 1: Collect Training Data (30 words + silence)
    train_audio_dir = os.path.join(BASE_DATA_PATH, "train/audio")
    if not os.path.exists(train_audio_dir):
        raise FileNotFoundError(f"Missing training audio directory: {train_audio_dir}")

    # Get the 30 word folders (excluding those starting with '_')
    word_labels = sorted([
        d for d in os.listdir(train_audio_dir)
        if os.path.isdir(os.path.join(train_audio_dir, d)) and not d.startswith('_')
    ])
    
    train_records = []
    
    # Load word command files
    for label in word_labels:
        folder_path = os.path.join(train_audio_dir, label)
        # Use sorted glob for deterministic file ordering
        files = sorted(glob(os.path.join(folder_path, "*.wav")))
        for f in files:
            # Extract speaker ID: os.path.basename(file).split('_nohash_')[0]
            speaker_id = os.path.basename(f).split('_nohash_')[0]
            train_records.append({
                'file_path': f,
                'label': label,
                'speaker_id': speaker_id
            })
    
    # Load silence samples from prepared directories
    silence_dirs = ["prepared_silence", "prepared_silence_v2"]
    for sd in silence_dirs:
        sd_path = os.path.join(BASE_DATA_PATH, sd)
        if os.path.exists(sd_path):
            files = sorted(glob(os.path.join(sd_path, "*.wav")))
            for f in files:
                speaker_id = os.path.basename(f).split('_nohash_')[0]
                train_records.append({
                    'file_path': f,
                    'label': 'silence',
                    'speaker_id': speaker_id
                })
    
    df_train = pd.DataFrame(train_records)
    print(f"Loaded {len(df_train)} training samples across {df_train['label'].nunique()} classes.")

    # Step 2: Collect Test Data
    test_audio_dir = os.path.join(BASE_DATA_PATH, "test/audio")
    if not os.path.exists(test_audio_dir):
        raise FileNotFoundError(f"Missing test audio directory: {test_audio_dir}")
    
    test_files = sorted(glob(os.path.join(test_audio_dir, "*.wav")))
    test_records = []
    for f in test_files:
        test_records.append({
            'file_path': f,
            'fname': os.path.basename(f)
        })
    
    df_test = pd.DataFrame(test_records)
    print(f"Loaded {len(df_test)} test samples.")

    # Step 3: Apply validation_mode subsetting if enabled
    if validation_mode:
        print("Applying validation_mode: subsetting datasets to 200 rows.")
        # Training subset: use simple sample (random_state for reproducibility)
        if len(df_train) > 200:
            df_train = df_train.sample(n=200, random_state=42).reset_index(drop=True)
        # Test subset: use simple sample
        if len(df_test) > 200:
            df_test = df_test.sample(n=200, random_state=42).reset_index(drop=True)

    # Step 4: Final formatting
    # X_train: DataFrame with file paths and speaker metadata for splitting
    X_train = df_train[['file_path', 'speaker_id']]
    # y_train: Series with ground truth labels
    y_train = df_train['label']
    # X_test: DataFrame with file paths
    X_test = df_test[['file_path']]
    # test_ids: Series with filenames for submission mapping
    test_ids = df_test['fname']

    print("Data loading complete.")
    return X_train, y_train, X_test, test_ids