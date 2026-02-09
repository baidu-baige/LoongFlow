import numpy as np
import pandas as pd
import torch
import torchaudio
import torchaudio.transforms as T
from multiprocessing import Pool
from typing import Tuple

# Standard parameters defined by the technical specification
SR = 44100
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 441
DURATION = 10.0
TARGET_SAMPLES = int(SR * DURATION)

def _process_audio_file(args):
    """
    Worker function to process a single audio file into a Log-Mel Spectrogram.
    Handles loading, mono conversion, resampling, cropping/padding, and transformation.
    """
    path, mode = args
    
    # 1. Load waveform (torchaudio is efficient for WAV)
    waveform, sr = torchaudio.load(path)
    
    # 2. Ensure Mono: Freesound 2019 clips are mono, but we handle stereo just in case.
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
        
    # 3. Resample: Ensuring consistent 44100 Hz sample rate
    if sr != SR:
        resampler = T.Resample(sr, SR)
        waveform = resampler(waveform)
        
    waveform = waveform.squeeze(0)
    num_samples = waveform.shape[0]
    
    # 4. Temporal Windowing (10.0 seconds / 441,000 samples)
    if num_samples > TARGET_SAMPLES:
        if mode == 'random':
            # Random temporal cropping for training data
            start = np.random.randint(0, num_samples - TARGET_SAMPLES + 1)
        else:
            # Center cropping for validation and test data
            start = (num_samples - TARGET_SAMPLES) // 2
        waveform = waveform[start : start + TARGET_SAMPLES]
    elif num_samples < TARGET_SAMPLES:
        # Zero padding for shorter clips
        waveform = torch.nn.functional.pad(waveform, (0, TARGET_SAMPLES - num_samples))
        
    # 5. Vision-ready transformation (Log-Mel Spectrogram)
    # n_fft=2048, hop_length=441, n_mels=128 gives (128, 1001) for 441,000 samples
    mel_transform = T.MelSpectrogram(
        sample_rate=SR,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        center=True,
        pad_mode="reflect",
        power=2.0
    )
    db_transform = T.AmplitudeToDB()
    
    spec = mel_transform(waveform.unsqueeze(0))
    log_mel_spec = db_transform(spec)
    
    # Return as float32 numpy array for memory efficiency and downstream compatibility
    return log_mel_spec.squeeze(0).numpy().astype(np.float32)

def preprocess(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_val: pd.DataFrame,
    X_test: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Transforms raw audio metadata into model-ready Log-Mel Spectrogram tensors.
    Utilizes multi-core processing to handle large audio datasets efficiently.
    """
    print(f"Preprocessing started: {len(X_train)} train, {len(X_val)} val, {len(X_test)} test samples.")
    
    # Prepare task arguments for parallel execution
    # Training uses random cropping; Validation and Test use center cropping
    train_args = [(path, 'random') for path in X_train['path'].values]
    val_args = [(path, 'center') for path in X_val['path'].values]
    test_args = [(path, 'center') for path in X_test['path'].values]
    
    # Leverage all 36 available CPU cores for parallel audio processing
    with Pool(processes=36) as pool:
        print("Transforming training audio...")
        X_train_processed = pool.map(_process_audio_file, train_args)
        print("Transforming validation audio...")
        X_val_processed = pool.map(_process_audio_file, val_args)
        print("Transforming test audio...")
        X_test_processed = pool.map(_process_audio_file, test_args)
        
    # Convert lists to high-performance numpy arrays
    X_train_processed = np.stack(X_train_processed)
    X_val_processed = np.stack(X_val_processed)
    X_test_processed = np.stack(X_test_processed)
    
    # Ensure targets are float32 numpy arrays
    y_train_processed = y_train.values.astype(np.float32)
    y_val_processed = y_val.values.astype(np.float32)
    
    # Feature Scaling: Z-score normalization based on training statistics (Avoids data leakage)
    print("Fitting normalization on training set...")
    train_mean = np.mean(X_train_processed)
    train_std = np.std(X_train_processed)
    
    # Apply transformation to all sets
    eps = 1e-7
    X_train_processed = (X_train_processed - train_mean) / (train_std + eps)
    X_val_processed = (X_val_processed - train_mean) / (train_std + eps)
    X_test_processed = (X_test_processed - train_mean) / (train_std + eps)
    
    # Final Validation: Ensure no anomalies (NaN/Inf) were introduced
    if not np.isfinite(X_train_processed).all():
        raise ValueError("Preprocessing resulted in non-finite values in X_train.")
    if not np.isfinite(X_val_processed).all():
        raise ValueError("Preprocessing resulted in non-finite values in X_val.")
    if not np.isfinite(X_test_processed).all():
        raise ValueError("Preprocessing resulted in non-finite values in X_test.")
        
    print(f"Preprocessing complete. Output shape: {X_train_processed.shape[1:]}")
    return X_train_processed, y_train_processed, X_val_processed, y_val_processed, X_test_processed