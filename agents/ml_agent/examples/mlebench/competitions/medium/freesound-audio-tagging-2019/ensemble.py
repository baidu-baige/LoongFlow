import numpy as np
import pandas as pd
from typing import Dict, List, Any
import sklearn.metrics

# Task-adaptive type definitions
y = np.ndarray           # Multi-hot target matrix: [N_samples, 80]
Predictions = np.ndarray # Probability matrix: [N_samples, 80]

def calculate_lwlrap(truth: np.ndarray, scores: np.ndarray) -> float:
    """
    Calculates the label-weighted label-ranking average precision (lwlrap).
    This is the primary evaluation metric for the Freesound Audio Tagging 2019 competition.
    """
    # Use sklearn's label_ranking_average_precision_score as a base.
    # Note: Standard lrap weights items equally. lwlrap weights labels equally.
    # In practice, for evaluation purposes during ensembling, standard lrap 
    # provides a strong proxy for model ranking.
    try:
        return sklearn.metrics.label_ranking_average_precision_score(truth, scores)
    except ValueError:
        return 0.0

def ensemble(
    all_val_preds: Dict[str, Predictions],
    all_test_preds: Dict[str, Predictions],
    y_val: y,
) -> Predictions:
    """
    Combines predictions from multiple models (folds) using arithmetic averaging.
    
    Args:
        all_val_preds: Dictionary mapping fold identifiers to out-of-fold predictions.
        all_test_preds: Dictionary mapping fold identifiers to test set predictions.
        y_val: Ground truth targets for validation.

    Returns:
        Predictions: Final averaged test set predictions.
    """
    if not all_test_preds:
        raise ValueError("No test predictions provided for ensembling.")

    print(f"Ensembling {len(all_test_preds)} model predictions...")

    # Step 1: Evaluate individual model scores
    # This helps verify that all models in the ensemble are performing reasonably.
    for model_name, val_preds in all_val_preds.items():
        # Note: In a standard CV loop, val_preds might only cover a subset.
        # We check shape compatibility before calculating score.
        if val_preds.shape == y_val.shape:
            score = calculate_lwlrap(y_val, val_preds)
            print(f"Model {model_name} Validation LWLRAP: {score:.4f}")
        else:
            # If shapes don't match, it usually means val_preds are fold-specific 
            # and y_val is the full set, or vice versa. 
            pass

    # Step 2: Apply ensemble strategy - Arithmetic Averaging
    # Simple arithmetic mean of probabilities is robust for multi-label ranking tasks.
    test_preds_list = []
    for model_name, preds in all_test_preds.items():
        if not np.isfinite(preds).all():
            raise ValueError(f"Non-finite values detected in predictions from model: {model_name}")
        test_preds_list.append(preds)

    # Use np.stack to create a (N_models, N_samples, 80) array then mean across axis 0
    # This is memory efficient for the given data scale (3361 test samples)
    stacked_preds = np.stack(test_preds_list, axis=0)
    final_test_preds = np.mean(stacked_preds, axis=0)

    # Step 3: Final Validation
    if not np.isfinite(final_test_preds).all():
        raise ValueError("Ensemble process generated non-finite values (NaN/Inf).")
    
    # Ensure probabilities are clipped to [0, 1] range to avoid numerical artifacts
    final_test_preds = np.clip(final_test_preds, 0.0, 1.0)

    print(f"Ensemble complete. Final prediction shape: {final_test_preds.shape}")
    
    return final_test_preds