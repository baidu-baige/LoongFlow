import numpy as np
import librosa
from joblib import Parallel, delayed
from typing import Tuple

# Concrete type definitions for this bioacoustic task
# X: raw temporal signal [N, 4000] or transformed [N, 128, 63, 1]
# y: target labels [N]
X = np.ndarray
y = np.ndarray

def _transform_single_waveform(signal: np.ndarray) -> np.ndarray:
    """
    Transforms a single raw audio waveform into a PCEN-normalized Mel Spectrogram.
    Following parameters from the technical specification for whale upcall detection.
    """
    # Step 1: Compute Mel Spectrogram (Power)
    # Parameters: n_fft=256, hop_length=64, n_mels=128, fmin=10, fmax=500
    # Center=True (default) ensures we get the target temporal dimension of 63 frames from 4000 samples
    S = librosa.feature.melspectrogram(
        y=signal,
        sr=2000,
        n_fft=256,
        hop_length=64,
        n_mels=128,
        fmin=10,
        fmax=500,
        power=2.0
    )
    
    # Step 2: Per-Channel Energy Normalization (PCEN)
    # PCEN is highly effective at suppressing stationary noise and enhancing transient whale calls.
    # Parameters: gain=0.98, bias=2, power=0.5, time_constant=0.4
    pce = librosa.pcen(
        S,
        sr=2000,
        hop_length=64,
        gain=0.98,
        bias=2,
        power=0.5,
        time_constant=0.4,
        eps=1e-6
    )
    
    # Step 3: Add channel dimension and ensure float32
    # Resulting shape: (128, 63, 1)
    return pce[..., np.newaxis].astype(np.float32)

def preprocess(
    X_train: X,
    y_train: y,
    X_val: X,
    y_val: y,
    X_test: X
) -> Tuple[X, y, X, y, X]:
    """
    Transforms raw temporal audio data into 3D time-frequency tensors (PCEN-Mel Spectrograms).
    
    Implementation leverages multi-core CPU parallelism to handle high-throughput processing
    efficiently within the provided hardware context.
    """
    print(f"Preprocessing starts. Train size: {X_train.shape}, Val size: {X_val.shape}, Test size: {X_test.shape}")

    # Process Training Set
    print("Transforming training data...")
    X_train_processed = Parallel(n_jobs=-1)(
        delayed(_transform_single_waveform)(x) for x in X_train
    )
    X_train_processed = np.stack(X_train_processed)
    
    # Process Validation Set
    print("Transforming validation data...")
    X_val_processed = Parallel(n_jobs=-1)(
        delayed(_transform_single_waveform)(x) for x in X_val
    )
    X_val_processed = np.stack(X_val_processed)
    
    # Process Test Set
    print("Transforming test data...")
    X_test_processed = Parallel(n_jobs=-1)(
        delayed(_transform_single_waveform)(x) for x in X_test
    )
    X_test_processed = np.stack(X_test_processed)

    # Validate output integrity: No NaN or Infinity
    # We use in-place cleanup to be memory efficient on 440GB RAM
    for data_set in [X_train_processed, X_val_processed, X_test_processed]:
        if not np.isfinite(data_set).all():
            np.nan_to_num(data_set, copy=False, nan=0.0, posinf=1.0, neginf=0.0)

    # Verification of target shapes: (N, 128, 63, 1)
    assert X_train_processed.shape[1:] == (128, 63, 1), f"Incorrect shape: {X_train_processed.shape}"
    assert X_train_processed.shape[0] == y_train.shape[0], "Train sample mismatch"
    assert X_val_processed.shape[0] == y_val.shape[0], "Val sample mismatch"
    assert X_test_processed.shape[0] == X_test.shape[0], "Test completeness error"

    print(f"Preprocessing completed. Final shapes: Train{X_train_processed.shape}, Val{X_val_processed.shape}, Test{X_test_processed.shape}")

    return (
        X_train_processed,
        y_train,
        X_val_processed,
        y_val,
        X_test_processed
    )