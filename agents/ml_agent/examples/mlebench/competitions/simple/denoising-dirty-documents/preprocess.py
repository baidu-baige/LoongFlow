import numpy as np
import cv2
from typing import Tuple, List, Union
from concurrent.futures import ThreadPoolExecutor

# Task-adaptive type definitions
# X and y are represented as 4D numpy arrays (N, C, H, W) or lists of such arrays
# To handle differing spatial dimensions between patches and full images,
# we use 4D arrays for each set.
X = Union[np.ndarray, List[np.ndarray]]
y = Union[np.ndarray, List[np.ndarray]]


def preprocess(
    X_train: List[np.ndarray],
    y_train: List[np.ndarray],
    X_val: List[np.ndarray],
    y_val: List[np.ndarray],
    X_test: List[np.ndarray]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Transforms raw image data into model-ready format using patch extraction and augmentation.

    Training data is transformed into thousands of augmented 128x128 patches.
    Validation and Test data are padded to a consistent size (multiple of 32) for inference.
    """
    print("Starting preprocessing: Patch extraction and augmentation.")

    # 1. Configuration
    PATCH_SIZE = 128
    PATCHES_PER_IMAGE = 100
    U_NET_DIVISOR = 32
    RANDOM_SEED = 42
    np.random.seed(RANDOM_SEED)

    # 2. Determine target dimensions for full-image sets (Val/Test)
    # We find the maximum dimensions across all sets to ensure consistency
    all_imgs = X_train + X_val + X_test
    max_h = max(img.shape[0] for img in all_imgs)
    max_w = max(img.shape[1] for img in all_imgs)

    # Round up to the nearest multiple of U_NET_DIVISOR
    target_h = ((max_h + U_NET_DIVISOR - 1) // U_NET_DIVISOR) * U_NET_DIVISOR
    target_w = ((max_w + U_NET_DIVISOR - 1) // U_NET_DIVISOR) * U_NET_DIVISOR

    print(f"Target dimensions for padded images: {target_h}x{target_w}")

    # 3. Helper: Patch Extraction and Augmentation
    def extract_augmented_patches(img_dirty: np.ndarray, img_clean: np.ndarray) -> Tuple[
        List[np.ndarray], List[np.ndarray]]:
        h, w = img_dirty.shape

        # Ensure image is at least PATCH_SIZE
        if h < PATCH_SIZE or w < PATCH_SIZE:
            ph = max(0, PATCH_SIZE - h)
            pw = max(0, PATCH_SIZE - w)
            img_dirty = np.pad(img_dirty, ((0, ph), (0, pw)), mode='edge')
            img_clean = np.pad(img_clean, ((0, ph), (0, pw)), mode='edge')
            h, w = img_dirty.shape

        p_xs, p_ys = [], []
        for _ in range(PATCHES_PER_IMAGE):
            # Random crop
            r = np.random.randint(0, h - PATCH_SIZE + 1)
            c = np.random.randint(0, w - PATCH_SIZE + 1)

            px = img_dirty[r:r + PATCH_SIZE, c:c + PATCH_SIZE].copy()
            py = img_clean[r:r + PATCH_SIZE, c:c + PATCH_SIZE].copy()

            # Geometric Augmentations (applied to both X and y)
            # Horizontal Flip
            if np.random.random() > 0.5:
                px = np.flip(px, axis=0)
                py = np.flip(py, axis=0)
            # Vertical Flip
            if np.random.random() > 0.5:
                px = np.flip(px, axis=1)
                py = np.flip(py, axis=1)
            # Random Rotation (90, 180, 270)
            k = np.random.randint(0, 4)
            if k > 0:
                px = np.rot90(px, k)
                py = np.rot90(py, k)

            # Photometric Augmentations (applied to X only)
            # Random Brightness and Contrast
            alpha = np.random.uniform(0.85, 1.15)  # Contrast
            beta = np.random.uniform(-0.1, 0.1)  # Brightness
            px = np.clip(px * alpha + beta, 0.0, 1.0)

            # Add channel dimension (C, H, W)
            p_xs.append(px[np.newaxis, ...])
            p_ys.append(py[np.newaxis, ...])

        return p_xs, p_ys

    # 4. Helper: Padding for Inference
    def pad_to_target(img: np.ndarray, th: int, tw: int) -> np.ndarray:
        h, w = img.shape
        ph = th - h
        pw = tw - w
        # Pad right and bottom with edge values to maintain continuity
        padded = np.pad(img, ((0, ph), (0, pw)), mode='edge')
        return padded[np.newaxis, ...]  # Add channel dimension

    # 5. Process Training Set
    print(f"Generating {len(X_train) * PATCHES_PER_IMAGE} patches from {len(X_train)} training images...")
    X_train_processed_list = []
    y_train_processed_list = []

    # Parallel processing for speed
    with ThreadPoolExecutor(max_workers=36) as executor:
        results = list(executor.map(extract_augmented_patches, X_train, y_train))

    for p_xs, p_ys in results:
        X_train_processed_list.extend(p_xs)
        y_train_processed_list.extend(p_ys)

    X_train_processed = np.stack(X_train_processed_list).astype(np.float32)
    y_train_processed = np.stack(y_train_processed_list).astype(np.float32)

    # 6. Process Validation and Test Sets
    print("Processing validation and test sets...")
    X_val_processed = np.stack([pad_to_target(img, target_h, target_w) for img in X_val]).astype(np.float32)
    y_val_processed = np.stack([pad_to_target(img, target_h, target_w) for img in y_val]).astype(np.float32)
    X_test_processed = np.stack([pad_to_target(img, target_h, target_w) for img in X_test]).astype(np.float32)

    # 7. Final Validation
    sets = {
        "X_train": X_train_processed, "y_train": y_train_processed,
        "X_val": X_val_processed, "y_val": y_val_processed,
        "X_test": X_test_processed
    }

    for name, data in sets.items():
        if np.isnan(data).any() or np.isinf(data).any():
            raise ValueError(f"Preprocessing failed: NaN or Inf values detected in {name}")

    # Check alignment
    assert X_train_processed.shape[0] == y_train_processed.shape[0], "Train alignment mismatch"
    assert X_val_processed.shape[0] == y_val_processed.shape[0], "Val alignment mismatch"
    assert X_test_processed.shape[0] == len(X_test), "Test completeness mismatch"

    print(f"Preprocessing complete.")
    print(f"  Train: {X_train_processed.shape}")
    print(f"  Val:   {X_val_processed.shape}")
    print(f"  Test:  {X_test_processed.shape}")

    return X_train_processed, y_train_processed, X_val_processed, y_val_processed, X_test_processed