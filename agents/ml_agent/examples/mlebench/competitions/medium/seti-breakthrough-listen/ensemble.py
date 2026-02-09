import numpy as np
import pandas as pd
from typing import Dict, List, Any
from sklearn.metrics import roc_auc_score

BASE_DATA_PATH = "/mnt/pfs/loongflow/devmachine/h20-09/evolux/output/mlebench/seti-breakthrough-listen/prepared/public"
OUTPUT_DATA_PATH = "output/a429c40e-fe12-455c-8b05-ca9d732aabeb/1/executor/output"

# Task-adaptive type definitions for SETI Breakthrough Listen
y = np.ndarray           # Target vector (binary labels 0 or 1)
Predictions = np.ndarray # Probability predictions (0.0 to 1.0)

def ensemble(
    all_val_preds: Dict[str, Predictions],
    all_test_preds: Dict[str, Predictions],
    y_val: y,
) -> Predictions:
    """
    Combines predictions from multiple models (or folds) into a final robust output.
    Uses simple arithmetic averaging of probabilities as specified in the technical requirements.
    
    Args:
        all_val_preds (Dict[str, Predictions]): Dictionary mapping model/fold names to OOF predictions.
        all_test_preds (Dict[str, Predictions]): Dictionary mapping model/fold names to test predictions.
        y_val (y): Ground truth targets for validation performance monitoring.

    Returns:
        Predictions: Final combined test set predictions.
    """
    if not all_test_preds:
        raise ValueError("all_test_preds is empty. No predictions to ensemble.")

    print(f"Starting ensemble of {len(all_test_preds)} model/fold predictions...")

    # Step 1: Evaluate individual model scores and check consistency
    model_names = list(all_test_preds.keys())
    test_sample_counts = [len(all_test_preds[name]) for name in model_names]
    
    # Ensure all test predictions have the same dimensions
    if len(set(test_sample_counts)) > 1:
        raise ValueError(f"Inconsistent test prediction lengths: {dict(zip(model_names, test_sample_counts))}")

    # Evaluate validation performance if y_val and all_val_preds are aligned
    # In a typical K-fold workflow, all_val_preds contains the Out-of-Fold (OOF) predictions
    if y_val is not None and all_val_preds:
        individual_aucs = {}
        for name, val_p in all_val_preds.items():
            if len(val_p) == len(y_val):
                score = roc_auc_score(y_val, val_p)
                individual_aucs[name] = score
                print(f"Model/Fold '{name}' Validation ROC AUC: {score:.5f}")
            else:
                print(f"Warning: Skipping AUC for '{name}' due to length mismatch (Pred: {len(val_p)}, True: {len(y_val)})")
        
        # Calculate Ensemble Validation Score if possible
        valid_val_preds = [v for k, v in all_val_preds.items() if len(v) == len(y_val)]
        if valid_val_preds:
            ens_val_preds = np.mean(valid_val_preds, axis=0)
            ens_val_auc = roc_auc_score(y_val, ens_val_preds)
            print(f"Ensemble Validation ROC AUC: {ens_val_auc:.5f}")

    # Step 2: Apply ensemble strategy - Simple Arithmetic Averaging
    # Stack all test predictions into a matrix (N_models, N_samples)
    test_preds_matrix = np.stack(list(all_test_preds.values()), axis=0)
    
    # Compute the mean across the model axis (axis=0)
    final_test_predictions = np.mean(test_preds_matrix, axis=0)

    # Step 3: Final validation and cleanup
    if np.isnan(final_test_predictions).any():
        # Identify which source had NaNs to aid debugging
        for name, preds in all_test_preds.items():
            if np.isnan(preds).any():
                raise ValueError(f"Model/Fold '{name}' contains NaN values in test predictions.")
        raise ValueError("Final ensemble output contains NaN values.")

    if np.isinf(final_test_predictions).any():
        raise ValueError("Final ensemble output contains Infinity values.")

    # Ensure probabilities are clipped to [0, 1] range to avoid numerical drift
    final_test_predictions = np.clip(final_test_predictions, 0.0, 1.0)

    print(f"Ensemble complete. Generated predictions for {len(final_test_predictions)} test samples.")
    
    return final_test_predictions