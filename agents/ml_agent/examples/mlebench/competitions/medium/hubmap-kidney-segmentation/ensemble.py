import numpy as np
import pandas as pd
import cv2
import gc
import os
from typing import Dict, List, Any
from joblib import Parallel, delayed

# Task-adaptive type definitions
# y: pd.Series containing paths to ground truth mask tiles
# Predictions: np.ndarray of shape (N, 1024, 1024)
y = pd.Series
Predictions = np.ndarray

def _load_mask_tile(path: str) -> np.ndarray:
    """Helper to load a mask tile from disk and convert to boolean."""
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Mask tile not found at path: {path}")
    return (mask > 127)

def ensemble(
    all_val_preds: Dict[str, Predictions],
    all_test_preds: Dict[str, Predictions],
    y_val: y,
) -> Predictions:
    """
    Consolidates multi-fold model outputs and TTA into a single prediction map.
    Uses probability averaging and searches for the optimal threshold on validation data.
    """
    print("Stage 5: Starting Ensemble (Multi-fold & TTA averaging)...")

    # --- Step 1: Average Validation Predictions ---
    if not all_val_preds:
        raise ValueError("all_val_preds is empty. Ensure models were trained and predicted.")

    val_model_names = list(all_val_preds.keys())
    num_val_models = len(val_model_names)
    print(f"Averaging {num_val_models} validation models/TTAs...")

    avg_val_preds = None
    for name in val_model_names:
        preds = all_val_preds[name].astype(np.float32)
        if avg_val_preds is None:
            avg_val_preds = preds
        else:
            avg_val_preds += preds
        # Free memory of individual prediction if possible (dictionary may still hold reference)
        del preds
    
    avg_val_preds /= num_val_models
    gc.collect()

    # --- Step 2: Load Validation Ground Truth ---
    print(f"Loading {len(y_val)} ground truth mask tiles for threshold optimization...")
    # Use parallel processing to speed up I/O
    y_true_list = Parallel(n_jobs=36)(delayed(_load_mask_tile)(p) for p in y_val)
    y_true = np.stack(y_true_list)
    del y_true_list
    gc.collect()

    # Flatten for efficient Dice calculation
    y_true_flat = y_true.reshape(-1)
    avg_val_preds_flat = avg_val_preds.reshape(-1)

    # --- Step 3: Threshold Optimization ---
    print("Searching for optimal threshold in range [0.4, 0.5]...")
    best_th = 0.5
    best_dice = -1.0
    
    # Search threshold with 0.01 granularity
    thresholds = np.linspace(0.4, 0.5, 11)
    
    for th in thresholds:
        preds_bin = (avg_val_preds_flat > th)
        
        intersection = np.logical_and(y_true_flat, preds_bin).sum()
        sum_true = y_true_flat.sum()
        sum_pred = preds_bin.sum()
        
        if (sum_true + sum_pred) == 0:
            dice = 1.0
        else:
            dice = 2.0 * intersection / (sum_true + sum_pred)
            
        print(f"  Threshold {th:.2f} | Dice: {dice:.4f}")
        
        if dice > best_dice:
            best_dice = dice
            best_th = th
            
    print(f"Optimal threshold found: {best_th:.2f} (Dice: {best_dice:.4f})")

    # Cleanup validation data to save RAM
    del y_true, y_true_flat, avg_val_preds, avg_val_preds_flat
    gc.collect()

    # --- Step 4: Average Test Predictions ---
    if not all_test_preds:
        raise ValueError("all_test_preds is empty.")

    test_model_names = list(all_test_preds.keys())
    num_test_models = len(test_model_names)
    print(f"Averaging {num_test_models} test models/TTAs...")

    avg_test_preds = None
    for name in test_model_names:
        preds = all_test_preds[name].astype(np.float32)
        if avg_test_preds is None:
            avg_test_preds = preds
        else:
            avg_test_preds += preds
        del preds

    avg_test_preds /= num_test_models
    gc.collect()

    # --- Step 5: Apply Optimal Threshold ---
    print(f"Applying threshold {best_th:.2f} to test predictions...")
    # Convert to uint8 binary mask as per Technical Specification
    final_test_masks = (avg_test_preds > best_th).astype(np.uint8)

    # Sanity checks
    if np.isnan(final_test_masks).any() or np.isinf(final_test_masks).any():
        raise ValueError("Ensemble output contains NaN or Infinity values.")

    print(f"Ensemble complete. Final Test Shape: {final_test_masks.shape}")
    
    del avg_test_preds
    gc.collect()

    return final_test_masks