import pandas as pd
import numpy as np
import scipy.stats
import scipy.signal
from typing import Tuple, Dict, Union, Any
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Define types for the volcanic eruption prediction task
X_Dict = Dict[Union[int, str], pd.DataFrame]
X_Tabular = pd.DataFrame
y_Series = pd.Series

def _extract_features_single(segment_id: Union[int, str], df: pd.DataFrame) -> Dict[str, Any]:
    """
    Extracts time-domain and frequency-domain features from a single seismic segment.
    """
    feats = {'segment_id': segment_id}
    
    # Iterate through all 10 sensors
    for col in df.columns:
        v = df[col].values
        
        # 4. Missing Values: count NaNs per sensor
        nan_count = np.isnan(v).sum()
        feats[f'{col}_nan_count'] = float(nan_count)
        
        # Pre-fill NaNs with 0 for all computations as per specification
        v_filled = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 1. Time-domain: mean, std, var, min, max, skew, kurtosis
        feats[f'{col}_mean'] = float(np.mean(v_filled))
        feats[f'{col}_std'] = float(np.std(v_filled))
        feats[f'{col}_var'] = float(np.var(v_filled))
        feats[f'{col}_min'] = float(np.min(v_filled))
        feats[f'{col}_max'] = float(np.max(v_filled))
        feats[f'{col}_skew'] = float(scipy.stats.skew(v_filled))
        feats[f'{col}_kurtosis'] = float(scipy.stats.kurtosis(v_filled))
        
        # Quantiles (0.01, 0.05, 0.1, 0.9, 0.95, 0.99)
        qs = [0.01, 0.05, 0.1, 0.9, 0.95, 0.99]
        q_vals = np.quantile(v_filled, qs)
        for q, q_val in zip(qs, q_vals):
            feats[f'{col}_q{q}'] = float(q_val)
        
        # 2. Rolling Stats: Mean and std of rolling windows (size 100, 1000)
        s_filled = pd.Series(v_filled)
        for w in [100, 1000]:
            roll_mean = s_filled.rolling(window=w).mean()
            roll_std = s_filled.rolling(window=w).std()
            feats[f'{col}_roll_mean_mean_{w}'] = float(roll_mean.mean())
            feats[f'{col}_roll_mean_std_{w}'] = float(roll_mean.std())
            feats[f'{col}_roll_std_mean_{w}'] = float(roll_std.mean())
            feats[f'{col}_roll_std_std_{w}'] = float(roll_std.std())
            
        # 3. Frequency-domain: FFT and Welch
        # FFT computation (fillna 0 already done)
        fft_complex = np.fft.rfft(v_filled)
        fft_mag = np.abs(fft_complex)
        
        # Top 5 dominant frequencies (indices and magnitudes)
        top_5_indices = np.argsort(fft_mag)[-5:]
        for i, idx in enumerate(reversed(top_5_indices)):
            feats[f'{col}_fft_top_freq_{i}'] = float(idx)
            feats[f'{col}_fft_top_mag_{i}'] = float(fft_mag[idx])
            
        # Spectral entropy using Shannon entropy of the power spectrum
        psd = fft_mag**2
        psd_norm = psd / (np.sum(psd) + 1e-12)
        entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-12))
        feats[f'{col}_spectral_entropy'] = float(entropy)
        
        # PSD using Welch's method (binned energy into 10 bins)
        _, pxx = scipy.signal.welch(v_filled, nperseg=256)
        binned_pxx = np.array_split(pxx, 10)
        for i, b in enumerate(binned_pxx):
            feats[f'{col}_welch_bin_{i}'] = float(np.sum(b))
            
    return feats

def preprocess(
    X_train: X_Dict,
    y_train: y_Series,
    X_val: X_Dict,
    y_val: y_Series,
    X_test: X_Dict
) -> Tuple[X_Tabular, y_Series, X_Tabular, y_Series, X_Tabular]:
    """
    Transforms raw seismic signal dictionaries into model-ready tabular feature matrices.
    """
    print("Preprocess: Starting high-dimensional feature extraction...")
    
    def _process_data_dict(data_dict: X_Dict) -> pd.DataFrame:
        """Helper to parallelize feature extraction over a dictionary of segments."""
        # Use all 36 available cores
        results = Parallel(n_jobs=-1)(
            delayed(_extract_features_single)(sid, df) for sid, df in data_dict.items()
        )
        # Convert list of dicts to DataFrame and set index to segment_id
        df_feats = pd.DataFrame(results).set_index('segment_id')
        # Clean up any potential NaNs/Infs produced by statistical functions
        df_feats.fillna(0, inplace=True)
        df_feats.replace([np.inf, -np.inf], 0, inplace=True)
        return df_feats

    # Extract features for all sets
    X_train_raw = _process_data_dict(X_train).reindex(y_train.index)
    X_val_raw = _process_data_dict(X_val).reindex(y_val.index)
    
    # Ensure test set preserves all unique identifiers
    test_ids = list(X_test.keys())
    X_test_raw = _process_data_dict(X_test).reindex(test_ids)

    print(f"Preprocess: Extraction complete. Found {X_train_raw.shape[1]} features.")

    # Initialize transformers: fit ONLY on training data to prevent leakage
    # We use a constant imputer as a final safety layer and a standard scaler for model readiness
    imputer = SimpleImputer(strategy='constant', fill_value=0)
    scaler = StandardScaler()
    
    # Fit and transform training set
    X_train_arr = imputer.fit_transform(X_train_raw)
    X_train_arr = scaler.fit_transform(X_train_arr)
    
    # Transform validation and test sets
    X_val_arr = imputer.transform(X_val_raw)
    X_val_arr = scaler.transform(X_val_arr)
    
    X_test_arr = imputer.transform(X_test_raw)
    X_test_arr = scaler.transform(X_test_arr)
    
    # Reconstruct DataFrames to maintain index and column metadata
    X_train_proc = pd.DataFrame(X_train_arr, index=X_train_raw.index, columns=X_train_raw.columns)
    X_val_proc = pd.DataFrame(X_val_arr, index=X_val_raw.index, columns=X_val_raw.columns)
    X_test_proc = pd.DataFrame(X_test_arr, index=X_test_raw.index, columns=X_test_raw.columns)
    
    # Final check for alignment and data quality
    assert len(X_train_proc) == len(y_train)
    assert len(X_val_proc) == len(y_val)
    assert not X_train_proc.isna().any().any(), "NaN values found in processed training data"
    assert np.isfinite(X_train_proc.values).all(), "Inf values found in processed training data"
    
    print("Preprocess: Completed successfully.")
    return X_train_proc, y_train, X_val_proc, y_val, X_test_proc