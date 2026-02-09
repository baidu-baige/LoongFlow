import os
import gc
import cv2
import numpy as np
import pandas as pd
import tifffile
from typing import Dict, Tuple

# Import component functions
from load_data import load_data
from preprocess import preprocess
from get_splitter import get_splitter
from train_and_predict import PREDICTION_ENGINES
from ensemble import ensemble

BASE_DATA_PATH = "/mnt/pfs/loongflow/devmachine/h20-01/evolux/output/mlebench/hubmap-kidney-segmentation/prepared/public"
OUTPUT_DATA_PATH = "output/dfecbcec-c1a9-41b2-a54c-960bf09a6314/1/executor/output"

def rle_encode(img: np.ndarray) -> str:
    """
    Encodes a binary mask into HuBMAP RLE format (column-major).
    """
    pixels = img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def get_tiff_shape(path: str) -> Tuple[int, int]:
    """
    Retrieves the height and width of a TIFF image efficiently.
    """
    with tifffile.TiffFile(path) as tif:
        shape = tif.pages[0].shape
        if len(shape) == 3:
            # Handle (C, H, W) vs (H, W, C)
            if shape[0] < 10: 
                return shape[1], shape[2]
            else:
                return shape[0], shape[1]
        return shape[:2]

def remove_small_objects(mask: np.ndarray, min_size: int = 500) -> np.ndarray:
    """
    Removes small connected components from the binary mask.
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] < min_size:
            mask[labels == i] = 0
    return mask

def workflow() -> dict:
    """
    Orchestrates the complete end-to-end machine learning pipeline in production mode.
    """
    print("Initializing Workflow Stage...")
    
    # 1. Load full dataset
    print("Step 1: Loading complete dataset...")
    X_train_raw, y_train_raw, X_test_raw, test_ids = load_data(validation_mode=False)
    
    # 2. Set up data splitting strategy
    print("Step 2: Initializing data splitting strategy...")
    splitter = get_splitter(X_train_raw, y_train_raw)
    
    # 3. Cross-Validation Loop
    all_val_preds_list = []
    all_y_val_paths_list = []
    all_test_preds_list = []
    X_test_processed_metadata = None
    
    engine_fn = PREDICTION_ENGINES["unetplusplus_efficientnetb7"]
    
    for fold, (train_idx, val_idx) in enumerate(splitter.split(X_train_raw, y_train_raw)):
        print(f"\n--- Starting Fold {fold + 1} / {splitter.get_n_splits()} ---")
        
        # Split data for this fold
        X_tr, y_tr = X_train_raw.iloc[train_idx], y_train_raw.iloc[train_idx]
        X_va, y_va = X_train_raw.iloc[val_idx], y_train_raw.iloc[val_idx]
        
        # Preprocess (Tiling)
        X_tr_p, y_tr_p, X_va_p, y_va_p, X_te_p = preprocess(X_tr, y_tr, X_va, y_va, X_test_raw)
        
        # Train and Predict
        val_probs, test_probs = engine_fn(X_tr_p, y_tr_p, X_va_p, y_va_p, X_te_p)
        
        # Store fold results
        all_val_preds_list.append(val_probs)
        all_y_val_paths_list.append(y_va_p)
        all_test_preds_list.append(test_probs)
        
        if X_test_processed_metadata is None:
            X_test_processed_metadata = X_te_p
            
        # Clean up fold specific data structures
        del X_tr, y_tr, X_va, y_va, X_tr_p, y_tr_p, X_va_p, y_va_p, X_te_p
        gc.collect()

    # 4. Consolidate and Ensemble
    print("\nStep 4: Consolidating predictions and running ensemble...")
    
    # Concatenate OOF predictions and ground truth paths
    # Note: Each fold handles a unique subset of the training data as validation
    oof_probs = np.concatenate(all_val_preds_list, axis=0)
    y_val_all_paths = pd.concat(all_y_val_paths_list, axis=0).reset_index(drop=True)
    
    # Prepare dictionaries for ensemble function
    # ensemble() averages all entries in the dicts
    val_preds_dict = {"combined_oof": oof_probs}
    test_preds_dict = {f"fold_{i}": p for i, p in enumerate(all_test_preds_list)}
    
    # Get final binary masks for test tiles using optimized threshold
    final_test_masks = ensemble(val_preds_dict, test_preds_dict, y_val_all_paths)
    
    # 5. Compute Prediction Statistics
    print("Step 5: Calculating prediction statistics...")
    avg_test_probs = np.mean(all_test_preds_list, axis=0)
    
    prediction_stats = {
        "oof": {
            "mean": float(np.mean(oof_probs)),
            "std": float(np.std(oof_probs)),
            "min": float(np.min(oof_probs)),
            "max": float(np.max(oof_probs)),
        },
        "test": {
            "mean": float(np.mean(avg_test_probs)),
            "std": float(np.std(avg_test_probs)),
            "min": float(np.min(avg_test_probs)),
            "max": float(np.max(avg_test_probs)),
        }
    }
    
    # Clean up probability arrays to free RAM for reconstruction
    del oof_probs, avg_test_probs, all_val_preds_list, all_test_preds_list
    gc.collect()

    # 6. Reconstruct Full Masks and Generate Submission
    print("Step 6: Reconstructing full-size masks and encoding RLE...")
    submission_rows = []
    
    for img_id in test_ids:
        print(f" Processing image: {img_id}")
        # Get original image path and shape
        img_path = X_test_raw.loc[X_test_raw['id'] == img_id, 'img_path'].values[0]
        H, W = get_tiff_shape(img_path)
        
        # Initialize full-size binary mask
        full_mask = np.zeros((H, W), dtype=np.uint8)
        
        # Filter tiles belonging to this image
        img_tile_indices = X_test_processed_metadata.index[X_test_processed_metadata['id'] == img_id].tolist()
        
        for idx in img_tile_indices:
            row = X_test_processed_metadata.iloc[idx]
            tx, ty = int(row['tile_x']), int(row['tile_y'])
            tile_mask = final_test_masks[idx]
            
            # Place tile into full mask (using maximum to handle overlaps)
            # Tiles are guaranteed 1024x1024 by preprocess()
            full_mask[ty:ty+1024, tx:tx+1024] = np.maximum(full_mask[ty:ty+1024, tx:tx+1024], tile_mask)
        
        # Post-processing: Remove small noise objects
        full_mask = remove_small_objects(full_mask, min_size=500)
        
        # Encode to RLE
        rle = rle_encode(full_mask)
        submission_rows.append({'id': img_id, 'predicted': rle})
        
        # Memory cleanup
        del full_mask
        gc.collect()

    # 7. Finalize Deliverables
    print("Step 7: Finalizing submission file...")
    submission_df = pd.DataFrame(submission_rows)
    sub_file_path = os.path.join(OUTPUT_DATA_PATH, "submission.csv")
    submission_df.to_csv(sub_file_path, index=False)
    
    print(f"Workflow complete. Submission saved to {sub_file_path}")
    
    output_info = {
        "submission_file_path": sub_file_path,
        "prediction_stats": prediction_stats,
    }
    
    return output_info