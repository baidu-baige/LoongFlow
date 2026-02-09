import os
import gc
import numpy as np
import pandas as pd
import tifffile
import cv2
from typing import Tuple, Any
from joblib import Parallel, delayed

# Task-adaptive type definitions
X = pd.DataFrame  # Feature matrix: Metadata and patch paths/coordinates
y = pd.Series     # Target vector: Paths to mask patches

BASE_DATA_PATH = "/mnt/pfs/loongflow/devmachine/h20-01/evolux/output/mlebench/hubmap-kidney-segmentation/prepared/public"
OUTPUT_DATA_PATH = "output/dfecbcec-c1a9-41b2-a54c-960bf09a6314/1/executor/output"

# Technical Specification Parameters
TILE_SIZE = 1024
OVERLAP = 256
STRIDE = TILE_SIZE - OVERLAP
# Pre-calculated normalization stats from technical spec
NORM_MEAN = [0.654, 0.455, 0.622]
NORM_STD = [0.118, 0.160, 0.113]

def rle_decode(mask_rle: str, shape: Tuple[int, int]) -> np.ndarray:
    """
    Decodes a HuBMAP RLE string into a binary mask.
    HuBMAP RLE: numbered top to bottom, then left to right.
    """
    h, w = shape
    if pd.isna(mask_rle) or mask_rle == "":
        return np.zeros((h, w), dtype=np.uint8)
    
    s = str(mask_rle).split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    
    mask = np.zeros(h * w, dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        mask[lo:hi] = 1
    
    # Reshape column-wise (W rows of H pixels) then transpose to (H, W)
    return mask.reshape(w, h).T

def process_single_image(img_idx: int, X_df: pd.DataFrame, y_series: pd.Series, is_test: bool, split_name: str) -> list:
    """
    Processes a single image into tiles, saves them, and returns metadata.
    """
    row = X_df.iloc[img_idx]
    img_id = row['id']
    img_path = row['img_path']
    
    # Patch directories
    img_patch_dir = os.path.join(OUTPUT_DATA_PATH, f"patches_{split_name}", "images")
    mask_patch_dir = os.path.join(OUTPUT_DATA_PATH, f"patches_{split_name}", "masks")
    os.makedirs(img_patch_dir, exist_ok=True)
    os.makedirs(mask_patch_dir, exist_ok=True)
    
    tile_results = []
    
    try:
        # Load full image into RAM
        image = tifffile.imread(img_path)
        image = np.squeeze(image)
        
        # Standardize to (H, W, C)
        if image.ndim == 3:
            if image.shape[0] == 3: # (C, H, W)
                image = image.transpose(1, 2, 0)
        elif image.ndim == 2: # Grayscale
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            
        h, w = image.shape[:2]
        
        mask = None
        if not is_test:
            # y_series contains RLE for training/val images
            mask = rle_decode(y_series.iloc[img_idx], (h, w))
            
        for y_coord in range(0, h, STRIDE):
            for x_coord in range(0, w, STRIDE):
                # Ensure tile doesn't go out of bounds
                ty = max(0, min(y_coord, h - TILE_SIZE))
                tx = max(0, min(x_coord, w - TILE_SIZE))
                
                tile_img = image[ty:ty+TILE_SIZE, tx:tx+TILE_SIZE]
                
                # Filtering heuristic: contains tissue (not pure white/black)
                mean_val = tile_img.mean()
                is_tissue = (15 < mean_val < 240)
                
                has_mask = False
                tile_mask = None
                if not is_test:
                    tile_mask = mask[ty:ty+TILE_SIZE, tx:tx+TILE_SIZE]
                    if tile_mask.sum() > 0:
                        has_mask = True
                
                # Keep all tiles for test; keep tissue/mask tiles for training
                if is_test or is_tissue or has_mask:
                    patch_id = f"{img_id}_{ty}_{tx}"
                    img_patch_path = os.path.join(img_patch_dir, f"{patch_id}.jpg")
                    cv2.imwrite(img_patch_path, cv2.cvtColor(tile_img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
                    
                    mask_patch_path = "None"
                    if not is_test:
                        mask_patch_path = os.path.join(mask_patch_dir, f"{patch_id}.png")
                        cv2.imwrite(mask_patch_path, tile_mask * 255)
                    
                    # Store tile metadata
                    tile_info = row.to_dict()
                    tile_info.update({
                        'tile_path': img_patch_path,
                        'mask_path': mask_patch_path,
                        'tile_x': tx,
                        'tile_y': ty,
                        'tile_h': TILE_SIZE,
                        'tile_w': TILE_SIZE,
                        # Store normalization stats for the next stage
                        'norm_mean_r': NORM_MEAN[0],
                        'norm_mean_g': NORM_MEAN[1],
                        'norm_mean_b': NORM_MEAN[2],
                        'norm_std_r': NORM_STD[0],
                        'norm_std_g': NORM_STD[1],
                        'norm_std_b': NORM_STD[2]
                    })
                    tile_results.append(tile_info)
        
        del image
        if mask is not None: del mask
        gc.collect()
        
    except Exception as e:
        print(f"Error processing image {img_id}: {e}")
        raise e
        
    return tile_results

def preprocess(
    X_train: X,
    y_train: y,
    X_val: X,
    y_val: y,
    X_test: X
) -> Tuple[X, y, X, y, X]:
    """
    Transforms raw data into patched format for model training.
    """
    print("Stage 3: Preprocessing (Tiling & Metadata Extraction)")

    # Reset indices for consistent mapping
    X_train = X_train.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    X_val = X_val.reset_index(drop=True)
    y_val = y_val.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)

    # Parallel execution with memory-conscious worker count
    n_workers = 12
    
    print(f"Tiling Training Set ({len(X_train)} images)...")
    train_out = Parallel(n_jobs=n_workers)(delayed(process_single_image)(i, X_train, y_train, False, "train") for i in range(len(X_train)))
    
    print(f"Tiling Validation Set ({len(X_val)} images)...")
    val_out = Parallel(n_jobs=n_workers)(delayed(process_single_image)(i, X_val, y_val, False, "val") for i in range(len(X_val)))
    
    print(f"Tiling Test Set ({len(X_test)} images)...")
    test_out = Parallel(n_jobs=n_workers)(delayed(process_single_image)(i, X_test, None, True, "test") for i in range(len(X_test)))

    # Flatten results
    X_train_proc = pd.DataFrame([p for sub in train_out for p in sub])
    X_val_proc = pd.DataFrame([p for sub in val_out for p in sub])
    X_test_proc = pd.DataFrame([p for sub in test_out for p in sub])

    # Extract target paths and drop internal metadata columns to align structure
    y_train_processed = X_train_proc['mask_path'].copy()
    y_val_processed = X_val_proc['mask_path'].copy()
    
    # Columns to remove from feature matrices
    drop_cols = ['mask_path', 'img_path']
    X_train_processed = X_train_proc.drop(columns=drop_cols, errors='ignore')
    X_val_processed = X_val_proc.drop(columns=drop_cols, errors='ignore')
    X_test_processed = X_test_proc.drop(columns=drop_cols, errors='ignore')

    # Force identical columns across all sets
    X_val_processed = X_val_processed[X_train_processed.columns]
    X_test_processed = X_test_processed[X_train_processed.columns]

    # Fill any NaNs in metadata (e.g. from info_df missing values)
    X_train_processed = X_train_processed.fillna(-1)
    X_val_processed = X_val_processed.fillna(-1)
    X_test_processed = X_test_processed.fillna(-1)

    # Verify test set completeness
    original_ids = set(X_test['id'].unique())
    processed_ids = set(X_test_processed['id'].unique())
    if not original_ids.issubset(processed_ids):
        raise RuntimeError(f"Missing test images in output tiles: {original_ids - processed_ids}")

    print(f"Preprocessing complete. Tiles: Train={len(X_train_processed)}, Val={len(X_val_processed)}, Test={len(X_test_processed)}")
    
    return X_train_processed, y_train_processed, X_val_processed, y_val_processed, X_test_processed