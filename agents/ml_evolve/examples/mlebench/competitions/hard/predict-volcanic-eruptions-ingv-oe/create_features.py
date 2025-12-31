# -*- coding: utf-8 -*-
"""
LLM Generated Code
"""

from typing import Any, Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import kurtosis, skew

# Global cache to persist extracted features across cross-validation folds
_FEATURE_CACHE = {}


def _extract_features_single(path: str) -> dict:
    """
    Extracts time-domain and frequency-domain features from a single seismic segment CSV file.
    
    Args:
        path (str): Full path to the segment CSV file.
        
    Returns:
        dict: A dictionary of extracted features for the segment.
    """
    # Load raw segment data
    try:
        df = pd.read_csv(path)
    except Exception as e:
        # Propagate error as per requirements
        raise RuntimeError(f"Failed to read segment file at {path}: {e}")

    # Standardize to 10 sensors as per competition description
    sensor_cols = [f'sensor_{i}' for i in range(1, 11)]
    for c in sensor_cols:
        if c not in df.columns:
            df[c] = np.nan

    # Pre-processing: Linear interpolation and fill remaining NaNs with 0
    df = df[sensor_cols].interpolate(method='linear', limit_direction='both').fillna(0.0)

    feat_dict = {}
    quantiles_list = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]

    for col in sensor_cols:
        s = df[col].values

        # 1. Time-Domain Statistics
        feat_dict[f'{col}_mean'] = np.mean(s)
        feat_dict[f'{col}_std'] = np.std(s)
        feat_dict[f'{col}_min'] = np.min(s)
        feat_dict[f'{col}_max'] = np.max(s)
        feat_dict[f'{col}_skew'] = skew(s)
        feat_dict[f'{col}_kurt'] = kurtosis(s)
        q_vals = np.quantile(s, quantiles_list)
        for i, q in enumerate(quantiles_list):
            feat_dict[f'{col}_q{int(q * 100)}'] = q_vals[i]

        # 2. Spectral Features (FFT)
        fft_vals = np.abs(np.fft.rfft(s))
        feat_dict[f'{col}_fft_mean'] = np.mean(fft_vals)
        feat_dict[f'{col}_fft_std'] = np.std(fft_vals)
        feat_dict[f'{col}_fft_max'] = np.max(fft_vals)

        freqs = np.fft.rfftfreq(len(s))
        mag_sum = np.sum(fft_vals)
        if mag_sum > 0:
            # Spectral Centroid
            feat_dict[f'{col}_spec_centroid'] = np.sum(freqs * fft_vals) / mag_sum
            # Spectral Roll-off (85% energy)
            cum_mag = np.cumsum(fft_vals)
            roll_off_idx = np.searchsorted(cum_mag, 0.85 * mag_sum)
            roll_off_idx = min(roll_off_idx, len(freqs) - 1)
            feat_dict[f'{col}_spec_rolloff'] = freqs[roll_off_idx]
        else:
            feat_dict[f'{col}_spec_centroid'] = 0.0
            feat_dict[f'{col}_spec_rolloff'] = 0.0

        # 3. Rolling Statistics (Windows 100 and 1000)
        ps = pd.Series(s)
        for w in [100, 1000]:
            r_mean = ps.rolling(window=w).mean()
            r_std = ps.rolling(window=w).std()
            feat_dict[f'{col}_roll{w}_mean_mean'] = r_mean.mean()
            feat_dict[f'{col}_roll{w}_mean_std'] = r_mean.std()
            feat_dict[f'{col}_roll{w}_std_mean'] = r_std.mean()
            feat_dict[f'{col}_roll{w}_std_std'] = r_std.std()

        # 4. Differenced Features
        ds = np.diff(s)
        feat_dict[f'{col}_diff_mean'] = np.mean(ds)
        feat_dict[f'{col}_diff_std'] = np.std(ds)
        feat_dict[f'{col}_diff_min'] = np.min(ds)
        feat_dict[f'{col}_diff_max'] = np.max(ds)
        feat_dict[f'{col}_diff_skew'] = skew(ds)
        feat_dict[f'{col}_diff_kurt'] = kurtosis(ds)
        dq_vals = np.quantile(ds, quantiles_list)
        for i, q in enumerate(quantiles_list):
            feat_dict[f'{col}_diff_q{int(q * 100)}'] = dq_vals[i]

    # Clean non-finite values (NaN, Inf) resulting from stats on constant signals
    for k in feat_dict:
        if not np.isfinite(feat_dict[k]):
            feat_dict[k] = 0.0

    return feat_dict


def create_features(
    X_train: Any,
    y_train: Any,
    X_val: Any,
    y_val: Any,
    X_test: Any
) -> Tuple[Any, Any, Any, Any, Any]:
    """
    Creates features for a single fold of cross-validation using high-concurrency parallel processing.
    """
    # Identify all unique paths to process across train, val, and test sets
    all_paths = pd.concat([X_train['path'], X_val['path'], X_test['path']]).unique()

    # Check cache to avoid redundant computation across folds
    paths_to_compute = [p for p in all_paths if p not in _FEATURE_CACHE]

    # Utilize 112 CPU cores for parallel feature extraction
    if paths_to_compute:
        results = Parallel(n_jobs=-1)(
            delayed(_extract_features_single)(p) for p in paths_to_compute
        )
        for path, result in zip(paths_to_compute, results):
            _FEATURE_CACHE[path] = result

    # Map extracted features back to the original order of input DataFrames
    X_train_transformed = pd.DataFrame([_FEATURE_CACHE[p] for p in X_train['path']])
    X_val_transformed = pd.DataFrame([_FEATURE_CACHE[p] for p in X_val['path']])
    X_test_transformed = pd.DataFrame([_FEATURE_CACHE[p] for p in X_test['path']])

    # Restore indices to ensure row preservation
    X_train_transformed.index = X_train.index
    X_val_transformed.index = X_val.index
    X_test_transformed.index = X_test.index

    # Final validation for NaN and Infinity as per specification
    X_train_transformed = X_train_transformed.fillna(0.0).replace([np.inf, -np.inf], 0.0)
    X_val_transformed = X_val_transformed.fillna(0.0).replace([np.inf, -np.inf], 0.0)
    X_test_transformed = X_test_transformed.fillna(0.0).replace([np.inf, -np.inf], 0.0)

    # Return 5 non-None values: transformed features and original labels
    return X_train_transformed, y_train, X_val_transformed, y_val, X_test_transformed
