import numpy as np
from scipy.ndimage import rotate
from joblib import Parallel, delayed
from typing import Tuple

# Task-adaptive type definitions
X = np.ndarray  # 5D NumPy array (N, C, D, H, W)
y = np.ndarray  # 1D NumPy array (N,)

def preprocess(
    X_train: X,
    y_train: y,
    X_val: X,
    y_val: y,
    X_test: X
) -> Tuple[X, y, X, y, X]:
    """
    Transforms raw MRI volumetric data into model-ready format for a single fold.
    Standardizes intensities using percentile-based Z-score scaling and applies 3D augmentations.
    """
    print("Preprocess: Starting data transformation...")

    # Step 1: Modality Selection
    # Technical Spec: Output shape (N, 1, 64, 256, 256). 
    # We select the first modality (FLAIR) to comply with the (N, 1, ...) requirement.
    X_train = X_train[:, 0:1, :, :, :].astype(np.float32)
    X_val = X_val[:, 0:1, :, :, :].astype(np.float32)
    X_test = X_test[:, 0:1, :, :, :].astype(np.float32)
    
    # Step 2: Fit Normalization on Training Data ONLY
    # Method: 0-99th percentile clipping followed by Z-score scaling
    print("Preprocess: Computing normalization statistics on training set...")
    p0 = np.percentile(X_train, 0)
    p99 = np.percentile(X_train, 99)
    
    # Clip training data to calculate mean/std for Z-score
    X_train_clipped_temp = np.clip(X_train, p0, p99)
    mean_val = np.mean(X_train_clipped_temp)
    std_val = np.std(X_train_clipped_temp)
    del X_train_clipped_temp # Memory optimization: release temporary clipped array
    
    # Step 3: Apply Normalization to all sets
    def normalize_volume(vol: np.ndarray, p0: float, p99: float, mu: float, sigma: float) -> np.ndarray:
        vol = np.clip(vol, p0, p99)
        return (vol - mu) / (sigma + 1e-8)

    print("Preprocess: Normalizing volumes...")
    X_train = normalize_volume(X_train, p0, p99, mean_val, std_val)
    X_val = normalize_volume(X_val, p0, p99, mean_val, std_val)
    X_test = normalize_volume(X_test, p0, p99, mean_val, std_val)

    # Step 4: 3D Data Augmentation (Training set only)
    # Parameters: random rotation (+/- 10 deg), horizontal flipping, brightness/contrast
    def augment_3d_volume(vol: np.ndarray) -> np.ndarray:
        """
        Applies random 3D spatial and intensity augmentations to a single patient's volume.
        """
        # Random Rotation (+/- 10 degrees) on the axial plane (H, W)
        angle = np.random.uniform(-10, 10)
        # axes=(2, 3) corresponds to (H, W) in (C, D, H, W)
        vol = rotate(vol, angle, axes=(2, 3), reshape=False, order=1, mode='constant', cval=0.0)
        
        # Horizontal Flipping (W axis)
        if np.random.rand() > 0.5:
            vol = np.flip(vol, axis=3)
            
        # Brightness and Contrast Adjustments
        # Since data is Z-scored, alpha scales variance and beta shifts the mean
        alpha = np.random.uniform(0.9, 1.1)
        beta = np.random.uniform(-0.1, 0.1)
        vol = vol * alpha + beta
        
        return vol.astype(np.float32)

    print(f"Preprocess: Applying 3D augmentations to {len(X_train)} training samples using 36 cores...")
    # Leveraging all 36 cores for parallel augmentation processing
    X_train_list = Parallel(n_jobs=36)(
        delayed(augment_3d_volume)(X_train[i]) for i in range(len(X_train))
    )
    X_train = np.stack(X_train_list)
    
    # Step 5: Final Validation and Cleanup
    # Ensure no NaN or Inf values propagated through transformations
    for arr, name in [(X_train, "X_train"), (X_val, "X_val"), (X_test, "X_test")]:
        if np.isnan(arr).any() or np.isinf(arr).any():
            print(f"Warning: NaNs/Infs detected in {name}. Performing safe replacement.")
            np.nan_to_num(arr, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    # Validate output shapes and data alignment
    assert X_train.shape[1:] == (1, 64, 256, 256), f"X_train shape mismatch: {X_train.shape}"
    assert len(X_train) == len(y_train), "X_train and y_train row mismatch"
    assert len(X_val) == len(y_val), "X_val and y_val row mismatch"
    assert len(X_test) == len(X_test), "X_test completeness check failed"

    print(f"Preprocess complete. Final X_train shape: {X_train.shape}")
    return X_train, y_train, X_val, y_val, X_test