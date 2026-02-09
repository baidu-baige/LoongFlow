import numpy as np
import librosa
from sklearn.preprocessing import StandardScaler
from concurrent.futures import ProcessPoolExecutor
from typing import Tuple, List, Dict, Any

# Task-adaptive type definitions
X = List[np.ndarray]  # List of feature matrices (T_i, 555)
y = List[np.ndarray]  # List of target vectors (T_i,)

def interpolate_1d(x: np.ndarray) -> np.ndarray:
    """Fast linear interpolation for missing joint data."""
    idx = np.arange(len(x))
    good = ~np.isnan(x)
    if not np.any(good):
        return np.zeros_like(x)
    return np.interp(idx, idx[good], x[good])

def get_angles(skeleton: np.ndarray) -> np.ndarray:
    """
    Calculates joint angles for Elbows and Shoulders.
    skeleton shape: (T, 20, 3)
    Returns: (T, 4) array of angles in radians.
    """
    # Joint IDs: 2:ShoulderCenter, 4:ShoulderLeft, 5:ElbowLeft, 6:WristLeft, 
    # 8:ShoulderRight, 9:ElbowRight, 10:WristRight
    angle_triplets = [
        (4, 5, 6),  # Elbow Left
        (8, 9, 10), # Elbow Right
        (2, 4, 5),  # Shoulder Left (uses ShoulderCenter as ref)
        (2, 8, 9)   # Shoulder Right (uses ShoulderCenter as ref)
    ]
    
    angles_list = []
    for a_idx, b_idx, c_idx in angle_triplets:
        a, b, c = skeleton[:, a_idx], skeleton[:, b_idx], skeleton[:, c_idx]
        ba = a - b
        bc = c - b
        
        dot = np.sum(ba * bc, axis=-1)
        norm_ba = np.linalg.norm(ba, axis=-1)
        norm_bc = np.linalg.norm(bc, axis=-1)
        
        cosine_angle = dot / (norm_ba * norm_bc + 1e-8)
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        angles_list.append(angle[:, None])
        
    return np.concatenate(angles_list, axis=1)

def extract_features_and_mask(sample: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Transforms a single raw sample into a 555-dim multimodal signature.
    Features: 363 (Skeleton: Base, Vel, Acc) + 192 (Audio: Log-Mel, D1, D2) = 555.
    """
    skeleton_raw = sample['skeleton'].copy()  # Shape: (T, 20, 3)
    audio = sample['audio']
    fs = sample['fs']
    T = skeleton_raw.shape[0]
    
    # 0. Precise Label Masking (Target Generation)
    label_mask = np.zeros(T, dtype=np.int32)
    precise_labels = sample.get('precise_labels', [])
    for lbl in precise_labels:
        # Matlab indices are 1-based, convert to 0-based
        start_idx = max(0, lbl['begin'] - 1)
        end_idx = min(T, lbl['end'])
        label_mask[start_idx:end_idx] = lbl['id']
    
    if T == 0:
        return np.zeros((1, 555), dtype=np.float32), np.zeros(1, dtype=np.int32)

    # 1. Spatial Imputation
    missing_mask = np.all(skeleton_raw == 0, axis=2)
    skeleton_raw[missing_mask] = np.nan
    for j in range(20):
        for c in range(3):
            skeleton_raw[:, j, c] = interpolate_1d(skeleton_raw[:, j, c])

    # Save raw HipCenter for Global Motion extraction
    hip_center_raw = skeleton_raw[:, 0, :].copy()

    # 2. Spatial Normalization (User Invariance)
    # Translate: Set HipCenter (0) as origin
    skeleton = skeleton_raw - skeleton_raw[:, 0:1, :]
    
    # Rotate: Align torso (ShoulderLeft 4 to ShoulderRight 8) to X-axis
    v = skeleton[:, 8, :] - skeleton[:, 4, :]
    rot_angles = -np.arctan2(v[:, 2], v[:, 0])
    cos, sin = np.cos(rot_angles), np.sin(rot_angles)
    
    nx = skeleton[:, :, 0] * cos[:, None] + skeleton[:, :, 2] * sin[:, None]
    nz = -skeleton[:, :, 0] * sin[:, None] + skeleton[:, :, 2] * cos[:, None]
    skeleton_norm = skeleton.copy()
    skeleton_norm[:, :, 0] = nx
    skeleton_norm[:, :, 2] = nz

    # 3. Base Skeleton Features (121 dims)
    angles = get_angles(skeleton_norm) # 4 dims
    BONES = [
        (0,1), (1,2), (2,3), (2,4), (4,5), (5,6), (6,7), (2,8), (8,9), (9,10), (10,11), 
        (0,12), (12,13), (13,14), (14,15), (0,16), (16,17), (17,18), (18,19)
    ]
    bone_vecs = np.concatenate([skeleton_norm[:, b_end, :] - skeleton_norm[:, b_start, :] for b_start, b_end in BONES], axis=1) # 19*3 = 57 dims
    coords = skeleton_norm.reshape(T, 60) # 20*3 = 60 dims
    base_motion = np.concatenate([coords, angles, bone_vecs], axis=1) # 60 + 4 + 57 = 121

    # 4. Multi-Order Motion + Global Motion Injection (363 dims)
    velocity = np.diff(base_motion, axis=0, prepend=base_motion[0:1])
    acceleration = np.diff(velocity, axis=0, prepend=velocity[0:1])
    
    # Global Motion Injection: Replace HipCenter indices (0,1,2) with raw world diffs
    raw_vel = np.diff(hip_center_raw, axis=0, prepend=hip_center_raw[0:1])
    raw_acc = np.diff(raw_vel, axis=0, prepend=raw_vel[0:1])
    velocity[:, 0:3] = raw_vel
    acceleration[:, 0:3] = raw_acc
    
    skeleton_feats = np.concatenate([base_motion, velocity, acceleration], axis=1) # 363 dims

    # 5. Audio Features (192 dims: Log-Mel + Delta + Delta-Delta)
    if fs > 0 and len(audio) > 0:
        if audio.dtype != np.float32: 
            audio = audio.astype(np.float32)
        
        hop = fs // 20 # Align with 20Hz skeleton frame rate
        n_fft = 2048
        
        # Log-Mel Spectrogram
        mel = librosa.feature.melspectrogram(y=audio, sr=fs, n_mels=64, hop_length=hop, n_fft=n_fft)
        log_mel = librosa.power_to_db(mel, ref=np.max).T
        
        # Delta and Delta-Delta
        log_mel_d1 = librosa.feature.delta(log_mel, axis=0)
        log_mel_d2 = librosa.feature.delta(log_mel, order=2, axis=0)
        audio_feats = np.concatenate([log_mel, log_mel_d1, log_mel_d2], axis=1)
        
        # Synchronization alignment
        if audio_feats.shape[0] > T: 
            audio_feats = audio_feats[:T]
        elif audio_feats.shape[0] < T: 
            audio_feats = np.pad(audio_feats, ((0, T - audio_feats.shape[0]), (0, 0)), mode='edge')
    else:
        audio_feats = np.zeros((T, 192), dtype=np.float32)

    # 6. Final Concatenation (363 + 192 = 555)
    features = np.concatenate([skeleton_feats, audio_feats], axis=1)
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    
    return features.astype(np.float32), label_mask

def preprocess(
    X_train_raw: List[Dict[str, Any]],
    y_train_raw: List[List[int]],
    X_val_raw: List[Dict[str, Any]],
    y_val_raw: List[List[int]],
    X_test_raw: List[Dict[str, Any]]
) -> Tuple[X, y, X, y, X]:
    """
    Transforms raw multi-modal data into 555-dim feature matrices aligned at 20Hz.
    """
    print(f"Preprocessing {len(X_train_raw)} train, {len(X_val_raw)} val, {len(X_test_raw)} test samples...")

    # Parallel extraction using 36 cores
    with ProcessPoolExecutor(max_workers=36) as executor:
        print("Extracting features (Train)...")
        train_results = list(executor.map(extract_features_and_mask, X_train_raw))
        print("Extracting features (Val)...")
        val_results = list(executor.map(extract_features_and_mask, X_val_raw))
        print("Extracting features (Test)...")
        test_results = list(executor.map(extract_features_and_mask, X_test_raw))

    X_train_feats, y_train_processed = zip(*train_results)
    X_val_feats, y_val_processed = zip(*val_results)
    X_test_feats, _ = zip(*test_results)

    # Convert tuples to lists
    X_train_feats, y_train_processed = list(X_train_feats), list(y_train_processed)
    X_val_feats, y_val_processed = list(X_val_feats), list(y_val_processed)
    X_test_feats = list(X_test_feats)

    # Fit Scaler on training frames only to prevent leakage
    print("Fitting Scaler on training set...")
    scaler = StandardScaler()
    all_train_frames = np.concatenate(X_train_feats, axis=0)
    scaler.fit(all_train_frames)
    
    # Scale all datasets
    X_train_processed = [scaler.transform(f).astype(np.float32) for f in X_train_feats]
    X_val_processed = [scaler.transform(f).astype(np.float32) for f in X_val_feats]
    X_test_processed = [scaler.transform(f).astype(np.float32) for f in X_test_feats]

    # Integrity verification
    feat_dim = 555
    for name, dataset in [("train", X_train_processed), ("val", X_val_processed), ("test", X_test_processed)]:
        for i, seq in enumerate(dataset):
            if np.isnan(seq).any() or np.isinf(seq).any():
                raise ValueError(f"NaN/Inf detected in {name} sample {i}")
            if seq.shape[1] != feat_dim:
                raise ValueError(f"Dim mismatch in {name} sample {i}: expected {feat_dim}, got {seq.shape[1]}")

    print(f"Preprocessing complete. Final feature dimension: {feat_dim}")
    return X_train_processed, y_train_processed, X_val_processed, y_val_processed, X_test_processed