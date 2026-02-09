import numpy as np
import os
from typing import Tuple, List, Dict, Any
from joblib import Parallel, delayed
from PIL import Image
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import RFE
from lightgbm import LGBMClassifier

# Constants based on technical specification
N_BANDS = 16
N_BLOCKS = 10
N_ACOUSTIC = 38
N_GEO = 3
N_SEG_FEAT = N_ACOUSTIC + N_GEO
N_MOMENTS = 5
HIST_DIM = 100
N_FEATURES_TO_SELECT = 250
RFE_STEP = 20

def extract_all_features(sample: Dict[str, Any], site_encoder: LabelEncoder, sm_encoder: LabelEncoder) -> np.ndarray:
    """
    Constructs an enriched 572D feature vector for a single recording.
    """
    # 1. Ecological Features (7D)
    meta = sample.get('meta', {'site': 'Unknown', 'month': 0, 'hour': 0})
    site = meta.get('site', 'Unknown')
    month = meta.get('month', 0)
    hour = meta.get('hour', 0)
    site_month = f"{site}_{month}"
    
    is_spring = 1.0 if 3 <= month <= 6 else 0.0
    
    # Cyclic encoding for month (1-12) and hour (0-23)
    m_angle = 2.0 * np.pi * (month - 1) / 12.0 if month > 0 else 0.0
    h_angle = 2.0 * np.pi * hour / 24.0
    
    # Handle unseen labels during transform
    try:
        site_val = float(site_encoder.transform([site])[0])
    except (ValueError, KeyError):
        site_val = -1.0
        
    try:
        sm_val = float(sm_encoder.transform([site_month])[0])
    except (ValueError, KeyError):
        sm_val = -1.0

    eco_feat = np.array([
        site_val,
        np.sin(m_angle),
        np.cos(m_angle),
        np.sin(h_angle),
        np.cos(h_angle),
        sm_val,
        is_spring
    ], dtype=np.float32)

    # 2. Multiscale Spectral Texture (256D)
    # Global (96D: 16 bands * 6 stats)
    spectral_global = np.zeros(N_BANDS * 6, dtype=np.float32)
    # Localized (160D: 16 bands * 10 blocks)
    spectral_local = np.zeros(N_BANDS * N_BLOCKS, dtype=np.float32)
    
    bmp_path = sample.get('bmp_path', "")
    if bmp_path and os.path.exists(bmp_path):
        try:
            # Open and normalize spectrogram
            img = np.array(Image.open(bmp_path).convert('L'), dtype=np.float32) / 255.0
            h, w = img.shape
            bh = h // N_BANDS
            
            for i in range(N_BANDS):
                band = img[i*bh : (i+1)*bh, :] if i < N_BANDS-1 else img[i*bh:, :]
                # Divide band into 10 temporal blocks (1.0s each for 10s clip)
                blocks = np.array_split(band, N_BLOCKS, axis=1)
                
                b_means = np.array([np.mean(b) if b.size > 0 else 0.0 for b in blocks])
                b_stds = np.array([np.std(b) if b.size > 0 else 0.0 for b in blocks])
                
                # Global Stats (6 per band)
                g_mean = np.mean(band)
                g_std = np.std(band)
                d_mean = np.mean(np.abs(np.diff(b_means))) if len(b_means) > 1 else 0.0
                d_std = np.mean(np.abs(np.diff(b_stds))) if len(b_stds) > 1 else 0.0
                burstiness = np.std(b_means)
                temp_mod = np.mean(b_stds)
                
                spectral_global[i*6 : (i+1)*6] = [g_mean, g_std, d_mean, d_std, burstiness, temp_mod]
                
                # Localized Stats (10 means per band)
                spectral_local[i*N_BLOCKS : (i+1)*N_BLOCKS] = b_means
        except Exception:
            pass # Use zeros on failure

    # 3. Enriched Segment Stats (205D)
    seg = sample.get('seg', np.zeros((0, N_ACOUSTIC), dtype=np.float32))
    geo = sample.get('geo', np.zeros((0, N_GEO), dtype=np.float32))
    num_segments = seg.shape[0]
    
    seg_moments = np.zeros(N_SEG_FEAT * N_MOMENTS, dtype=np.float32)
    if num_segments > 0:
        inst_feat = np.concatenate([seg, geo], axis=1) # (K, 41)
        # Moments: Mean, Max, Std, Q50, Q95
        m_mean = np.mean(inst_feat, axis=0)
        m_max = np.max(inst_feat, axis=0)
        m_std = np.std(inst_feat, axis=0)
        m_q50 = np.median(inst_feat, axis=0)
        m_q95 = np.percentile(inst_feat, 95, axis=0)
        seg_moments = np.concatenate([m_mean, m_max, m_std, m_q50, m_q95])

    # 4. Bag Geometry (4D)
    total_area = np.sum(geo[:, 2]) if num_segments > 0 else 0.0
    # Density: Area / (Duration * Bandwidth)
    densities = geo[:, 2] / (geo[:, 0] * geo[:, 1] + 1e-9) if num_segments > 0 else np.array([0.0])
    avg_density = np.mean(densities) if num_segments > 0 else 0.0
    has_segments = 1.0 if num_segments > 0 else 0.0
    bag_geo = np.array([float(num_segments), total_area, avg_density, has_segments], dtype=np.float32)

    # 5. Histogram (100D)
    hist = sample.get('hist', np.zeros(HIST_DIM, dtype=np.float32))
    
    # Total Vector Construction (572D)
    full_vector = np.concatenate([
        eco_feat,           # 7
        spectral_global,    # 96
        spectral_local,     # 160
        seg_moments,        # 205
        bag_geo,            # 4
        hist                # 100
    ])
    
    return np.nan_to_num(full_vector, nan=0.0, posinf=0.0, neginf=0.0)

def preprocess(
    X_train: List[Dict[str, Any]],
    y_train: np.ndarray,
    X_val: List[Dict[str, Any]],
    y_val: np.ndarray,
    X_test: List[Dict[str, Any]]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Transforms raw data into model-ready 250D matrices using RFE and multiscale features.
    """
    print(f"Preprocessing {len(X_train)} train, {len(X_val)} val, {len(X_test)} test samples.")

    # Step 1: Fit Categorical Encoders on Training data
    site_le = LabelEncoder()
    unique_sites = list(set(s['meta']['site'] for s in X_train))
    site_le.fit(unique_sites + ['Unknown'])
    
    sm_le = LabelEncoder()
    unique_sm = list(set(f"{s['meta']['site']}_{s['meta']['month']}" for s in X_train))
    sm_le.fit(unique_sm + ['Unknown_0'])

    # Step 2: Parallel Feature Extraction (36 Cores)
    def extract_dataset(dataset, desc):
        print(f"Extracting 572D features for {desc} set...")
        features = Parallel(n_jobs=36)(
            delayed(extract_all_features)(s, site_le, sm_le) for s in dataset
        )
        return np.stack(features).astype(np.float32)

    X_train_raw = extract_dataset(X_train, "train")
    X_val_raw = extract_dataset(X_val, "val")
    X_test_raw = extract_dataset(X_test, "test")

    # Step 3: Standardization
    print("Standardizing feature space...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_val_scaled = scaler.transform(X_val_raw)
    X_test_scaled = scaler.transform(X_test_raw)

    # Step 4: Recursive Feature Selection (RFE)
    # Target for RFE: Presence of any bird (binary) to capture acoustic relevance
    y_train_binary = (y_train.sum(axis=1) > 0).astype(int)
    
    print(f"Pruning feature space from {X_train_scaled.shape[1]}D to {N_FEATURES_TO_SELECT}D using RFE...")
    lgbm = LGBMClassifier(
        n_estimators=50, 
        max_depth=3, 
        learning_rate=0.1, 
        importance_type='gain', 
        n_jobs=-1,
        verbose=-1,
        random_state=42
    )
    
    selector = RFE(
        estimator=lgbm, 
        n_features_to_select=N_FEATURES_TO_SELECT, 
        step=RFE_STEP, 
        verbose=1
    )
    
    X_train_pruned = selector.fit_transform(X_train_scaled, y_train_binary)
    X_val_pruned = selector.transform(X_val_scaled)
    X_test_pruned = selector.transform(X_test_scaled)

    # Step 5: Final Sanity Check and Cleaning
    X_train_final = np.nan_to_num(X_train_pruned.astype(np.float32))
    X_val_final = np.nan_to_num(X_val_pruned.astype(np.float32))
    X_test_final = np.nan_to_num(X_test_pruned.astype(np.float32))
    
    y_train_final = y_train.astype(np.float32)
    y_val_final = y_val.astype(np.float32)

    print(f"Preprocessing complete. Output shape: {X_train_final.shape}")
    return X_train_final, y_train_final, X_val_final, y_val_final, X_test_final