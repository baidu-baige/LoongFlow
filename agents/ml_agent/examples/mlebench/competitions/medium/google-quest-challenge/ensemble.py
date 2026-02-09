import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from typing import Dict, List, Any

# Task-adaptive type definitions
y = pd.DataFrame           # Target vector type (30 continuous labels)
Predictions = np.ndarray   # Model predictions type (N_samples, 30)

BASE_DATA_PATH = "/mnt/pfs/loongflow/devmachine/2-05/evolux/output/mlebench/google-quest-challenge/prepared/public"
OUTPUT_DATA_PATH = "output/aaa741b3-cb02-44fc-a666-dd434e563444/8/executor/output"

def ensemble(
    all_val_preds: Dict[str, Predictions],
    all_test_preds: Dict[str, Predictions],
    y_val: y,
) -> Predictions:
    """
    Combines predictions from multiple models and folds into a final robust output.
    
    Strategy:
    1. Arithmetic mean of all provided test predictions (averaging cross-validation folds).
    2. Monotonic column-wise Min-Max scaling to maximize Spearman rank distribution.
    3. Clipping to range [0, 1].
    
    Args:
        all_val_preds (Dict[str, Predictions]): Model/Fold names mapped to OOF predictions.
        all_test_preds (Dict[str, Predictions]): Model/Fold names mapped to test set predictions.
        y_val (y): Ground truth targets for validation performance tracking.

    Returns:
        Predictions: Final averaged and processed test set predictions (N_test, 30).
    """
    print("Execution Stage: ensemble")

    if not all_test_preds:
        raise ValueError("Critical Error: all_test_preds is empty. No predictions available for ensembling.")

    # Step 1: Evaluate individual model/fold scores (Diagnostic)
    # Spearman correlation is the primary metric. We calculate mean column-wise Spearman.
    print(f"Evaluating {len(all_val_preds)} individual validation prediction sets...")
    
    y_val_arr = y_val.values
    for name, val_preds in all_val_preds.items():
        # Check alignment with y_val for diagnostics. 
        # Note: In CV, val_preds might be a subset (single fold). 
        # If shape matches, we compute the score.
        if val_preds.shape == y_val_arr.shape:
            column_spearmans = []
            for i in range(y_val_arr.shape[1]):
                with np.errstate(divide='ignore', invalid='ignore'):
                    res = spearmanr(y_val_arr[:, i], val_preds[:, i]).correlation
                
                # Handle cases with constant predictions (Spearman returns NaN)
                if np.isnan(res):
                    res = 0.0
                column_spearmans.append(res)
            
            mean_spearman = np.mean(column_spearmans)
            print(f"Model/Fold '{name}' | Mean Spearman: {mean_spearman:.4f}")
        else:
            # Shape mismatch usually means val_preds is for a specific fold, not the whole dataset.
            # We skip detailed scoring to avoid index errors, as the main goal is ensembling test preds.
            print(f"Diagnostic: '{name}' val_preds shape {val_preds.shape} differs from y_val {y_val_arr.shape}. Skipping OOF score.")

    # Step 2: Apply Ensemble Strategy (Simple Arithmetic Mean)
    # The technical specification requires averaging fold-wise and model-wise predictions.
    # upstream 'train_and_predict' already averages RoBERTa and BERT (0.5 weight each).
    # Thus, simple mean of the entries in all_test_preds satisfies the weight requirements.
    
    num_sets = len(all_test_preds)
    test_preds_list = list(all_test_preds.values())
    target_shape = test_preds_list[0].shape
    
    print(f"Averaging test predictions from {num_sets} source sets. Target shape: {target_shape}")
    
    # Initialize accumulator with float64 to maintain precision
    ensemble_test_preds = np.zeros(target_shape, dtype=np.float64)
    
    for name, preds in all_test_preds.items():
        if preds.shape != target_shape:
            raise ValueError(f"Shape mismatch in test predictions for '{name}'. Expected {target_shape}, got {preds.shape}")
        ensemble_test_preds += preds.astype(np.float64)
    
    # Compute arithmetic mean
    ensemble_test_preds /= num_sets
    
    # Step 3: Post-processing (Column-wise Min-Max Scaling)
    # As per technical specification, stretching the rank distribution can boost Spearman scores.
    # Formula: (preds - min) / (max - min + 1e-9)
    print("Applying column-wise Min-Max scaling to optimize Spearman rank distribution...")
    
    p_min = ensemble_test_preds.min(axis=0)
    p_max = ensemble_test_preds.max(axis=0)
    
    # Avoid division by zero with epsilon 1e-9 as specified
    final_test_preds = (ensemble_test_preds - p_min) / (p_max - p_min + 1e-9)
    
    # Step 4: Clipping and Validation
    # Ensure all values are strictly in [0, 1] as per competition requirements
    final_test_preds = np.clip(final_test_preds, 0.0, 1.0).astype(np.float32)

    # Integrity Check for NaNs or Inf
    if np.isnan(final_test_preds).any() or np.isinf(final_test_preds).any():
        raise ValueError("Ensemble process generated invalid numeric values (NaN/Inf). Check input distributions.")

    print(f"Ensemble complete. Generated predictions for {final_test_preds.shape[0]} samples across {final_test_preds.shape[1]} labels.")
    
    return final_test_preds