import os
import aifc
import numpy as np
from joblib import Parallel, delayed
from typing import Tuple, Any

# Constants
BASE_DATA_PATH = "/mnt/pfs/loongflow/devmachine/h20-01/evolux/output/mlebench/the-icml-2013-whale-challenge-right-whale-redux/prepared/public"
OUTPUT_DATA_PATH = "output/6795de84-a7ab-443d-bbf5-03771db15966/1/executor/output"

# Task-adaptive type definitions
X = np.ndarray      # Feature matrix: [N, 4000] float32
y = np.ndarray      # Target vector: [N] int64
Ids = np.ndarray    # Identifier array: [M] string (filenames)

def _process_single_aif(file_path: str, target_len: int = 4000) -> np.ndarray:
    """
    Reads a single .aif file, normalizes it, and ensures it matches the target length.
    AIFF files in this dataset are 16-bit PCM, Big-Endian, 2000Hz.
    """
    try:
        with aifc.open(file_path, 'r') as f:
            n_frames = f.getnframes()
            frames = f.readframes(n_frames)
            
            # Convert bytes to numpy array. AIFF is Big-Endian ('>'). 
            # 16-bit signed integer format is 'i2'.
            waveform = np.frombuffer(frames, dtype='>i2').astype(np.float32)
            
            # Normalize waveform to range [-1.0, 1.0]
            if waveform.size > 0:
                waveform /= 32768.0
            
            # Apply padding or truncation to reach exactly target_len (4000 samples)
            if len(waveform) > target_len:
                waveform = waveform[:target_len]
            elif len(waveform) < target_len:
                waveform = np.pad(waveform, (0, target_len - len(waveform)), mode='constant')
            
            return waveform
    except Exception as e:
        raise RuntimeError(f"Failed to process audio file {file_path}: {e}")

def _find_aif_files(directory: str) -> list:
    """Recursively finds all .aif files in a directory."""
    aif_files = []
    for root, _, files in os.walk(directory):
        for f in files:
            if f.lower().endswith('.aif'):
                aif_files.append(os.path.join(root, f))
    return sorted(aif_files)

def load_data(validation_mode: bool = False) -> Tuple[X, y, X, Ids]:
    """
    Loads and prepares the Whale Challenge dataset.
    
    Args:
        validation_mode: If True, returns a small subset (up to 200 samples).
        
    Returns:
        X_train (np.ndarray): Training features (N_train, 4000)
        y_train (np.ndarray): Training labels (N_train,)
        X_test (np.ndarray): Test features (N_test, 4000)
        test_ids (np.ndarray): Test filenames for submission (N_test,)
    """
    print(f"Starting data loading from: {BASE_DATA_PATH}")
    
    train_root = os.path.join(BASE_DATA_PATH, "train_extracted")
    test_root = os.path.join(BASE_DATA_PATH, "test_extracted")
    
    # Locate all .aif files recursively to handle potential nesting
    all_train_paths = _find_aif_files(train_root)
    all_test_paths = _find_aif_files(test_root)
    
    if not all_train_paths or not all_test_paths:
        raise ValueError(f"No .aif audio files found. \nTrain root: {train_root} (found {len(all_train_paths)}) \nTest root: {test_root} (found {len(all_test_paths)})")

    # Apply validation subsetting if requested
    if validation_mode:
        print("Validation mode: selecting first 200 samples.")
        all_train_paths = all_train_paths[:200]
        all_test_paths = all_test_paths[:200]
    
    # Extract labels from training file names: _1.aif is whale, _0.aif is noise
    y_train = []
    for path in all_train_paths:
        fname = os.path.basename(path).lower()
        if fname.endswith("_1.aif"):
            y_train.append(1)
        elif fname.endswith("_0.aif"):
            y_train.append(0)
        else:
            # Fallback for unexpected naming, though specification implies these suffix patterns
            raise ValueError(f"Unexpected training filename format: {path}")
    y_train = np.array(y_train, dtype=np.int64)

    # Parallel loading of audio files
    print(f"Loading {len(all_train_paths)} training samples...")
    X_train_list = Parallel(n_jobs=-1)(
        delayed(_process_single_aif)(p) for p in all_train_paths
    )
    X_train = np.stack(X_train_list).astype(np.float32)

    print(f"Loading {len(all_test_paths)} test samples...")
    X_test_list = Parallel(n_jobs=-1)(
        delayed(_process_single_aif)(p) for p in all_test_paths
    )
    X_test = np.stack(X_test_list).astype(np.float32)

    # test_ids must be the filename (basename) for alignment with submission requirements
    test_ids = np.array([os.path.basename(p) for p in all_test_paths])

    # Final validation of shapes
    assert len(X_train) == len(y_train), f"X_train ({len(X_train)}) and y_train ({len(y_train)}) mismatch"
    assert len(X_test) == len(test_ids), f"X_test ({len(X_test)}) and test_ids ({len(test_ids)}) mismatch"

    print("Data loading completed.")
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    
    return X_train, y_train, X_test, test_ids