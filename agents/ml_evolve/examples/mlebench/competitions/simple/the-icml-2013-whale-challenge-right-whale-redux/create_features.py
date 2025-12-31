# -*- coding: utf-8 -*-
"""
LLM Generated Code
"""

from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torchaudio

# Define the data paths
BASE_DATA_PATH = "/root/workspace/evolux/output/mlebench/the-icml-2013-whale-challenge-right-whale-redux/prepared/public"
OUTPUT_DATA_PATH = "output/5872ea81-cdbf-4bc6-8da8-7d0e82d40021/2/executor/output"

# For type hinting, DT is assumed to be pandas DataFrame or Series
DT = pd.DataFrame | pd.Series


def create_features(
    X_train: DT,
    y_train: DT,
    X_val: DT,
    y_val: DT,
    X_test: DT
) -> Tuple[DT, DT, DT, DT, DT]:
    """
    Creates features for a single fold of cross-validation using GPU-accelerated Log-Mel Spectrograms.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize GPU-based transforms
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=2000,
        n_fft=512,
        hop_length=64,
        n_mels=128,
        f_min=30.0,
        f_max=500.0,
        center=True,
        pad_mode="reflect",
        power=2.0
    ).to(device)

    db_transform = torchaudio.transforms.AmplitudeToDB(stype='power', top_db=80.0).to(device)

    freq_masking = torchaudio.transforms.FrequencyMasking(freq_mask_param=15).to(device)
    time_masking = torchaudio.transforms.TimeMasking(time_mask_param=25).to(device)

    def process_dataset(df: DT, augment: bool = False) -> torch.Tensor:
        # Convert to tensor and move to GPU
        tensor = torch.tensor(df.values, device=device, dtype=torch.float32)

        if augment:
            # Waveform Augmentation: Gaussian Noise
            tensor += torch.randn_like(tensor) * 0.005
            # Waveform Augmentation: Random Time Shift
            for i in range(tensor.shape[0]):
                shift = np.random.randint(-100, 101)
                tensor[i] = torch.roll(tensor[i], shifts=shift, dims=0)

        # Spectrogram Transformation
        mel_spec = mel_transform(tensor)
        log_mel = db_transform(mel_spec)

        if augment:
            # SpecAugment: Applying masks individually per sample for diversity
            for i in range(log_mel.shape[0]):
                log_mel[i:i + 1] = freq_masking(log_mel[i:i + 1])
                log_mel[i:i + 1] = time_masking(log_mel[i:i + 1])

        # Flatten for model compatibility: (N, n_mels, time_steps) -> (N, n_mels * time_steps)
        return log_mel.reshape(log_mel.shape[0], -1)

    with torch.no_grad():
        # Step 1 & 2: Apply transformations to train, val, and test sets
        X_train_f = process_dataset(X_train, augment=True)
        X_val_f = process_dataset(X_val, augment=False)
        X_test_f = process_dataset(X_test, augment=False)

        # Step 3: Normalization (Standardization based on training data)
        mean = X_train_f.mean(dim=0, keepdim=True)
        std = X_train_f.std(dim=0, keepdim=True)

        # Avoid division by zero for constant features
        std[std == 0] = 1.0

        X_train_f = (X_train_f - mean) / std
        X_val_f = (X_val_f - mean) / std
        X_test_f = (X_test_f - mean) / std

        # Ensure no NaN or Infinity values
        X_train_f = torch.nan_to_num(X_train_f)
        X_val_f = torch.nan_to_num(X_val_f)
        X_test_f = torch.nan_to_num(X_test_f)

        # Step 4: Convert back to pandas DataFrames with consistent structure
        feat_cols = [f"mel_feat_{i}" for i in range(X_train_f.shape[1])]

        X_train_transformed = pd.DataFrame(
            X_train_f.cpu().numpy(),
            columns=feat_cols,
            index=X_train.index
        )
        X_val_transformed = pd.DataFrame(
            X_val_f.cpu().numpy(),
            columns=feat_cols,
            index=X_val.index
        )
        X_test_transformed = pd.DataFrame(
            X_test_f.cpu().numpy(),
            columns=feat_cols,
            index=X_test.index
        )

    return X_train_transformed, y_train, X_val_transformed, y_val, X_test_transformed
