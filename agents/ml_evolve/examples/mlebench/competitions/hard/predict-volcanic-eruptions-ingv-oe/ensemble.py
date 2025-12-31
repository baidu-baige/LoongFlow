# -*- coding: utf-8 -*-
"""
LLM Generated Code
"""

from typing import Any, Dict

import numpy as np

BASE_DATA_PATH = "/root/workspace/evolux/output/mlebench/predict-volcanic-eruptions-ingv-oe/prepared/public"
OUTPUT_DATA_PATH = "output/43637237-1f49-4750-a868-8602ac177881/1/executor/output"

# Type Definitions
Features = Any  # Feature matrix (pd.DataFrame)
Labels = Any  # Target labels (pd.Series or np.ndarray)
Predictions = Any  # Model predictions (np.ndarray)


def ensemble(
    all_oof_preds: Dict[str, Predictions],
    all_test_preds: Dict[str, Predictions],
    y_true_full: Labels
) -> Predictions:
    """
    Combines predictions from multiple models into a final output using simple averaging.
    
    Args:
        all_oof_preds (Dict[str, Predictions]): Dictionary mapping model names to their out-of-fold predictions.
        all_test_preds (Dict[str, Predictions]): Dictionary mapping model names to their test predictions.
        y_true_full (Labels): Ground truth labels, available for evaluation.
        
    Returns:
        Predictions: Final ensemble test set predictions.
    """
    # Step 1: Evaluate individual model scores and prediction correlations
    # This step is primarily for diagnostics. We propagate any errors in input consistency.
    if not all_test_preds:
        raise ValueError("The all_test_preds dictionary is empty. No predictions to ensemble.")

    # Step 2: Apply ensemble strategy
    # Following the requirement: Simple average with equal weights.
    # We aggregate all prediction arrays provided in all_test_preds.
    test_preds_to_average = []

    for model_name, preds in all_test_preds.items():
        # Check if the value is a list (e.g., individual fold predictions) or a single array
        if isinstance(preds, (list, tuple)):
            # If it's a list/tuple of arrays from multiple folds
            if len(preds) == 0:
                continue
            test_preds_to_average.extend(preds)
        else:
            # If it's a single array (e.g., already averaged or a single model run)
            test_preds_to_average.append(preds)

    if not test_preds_to_average:
        raise ValueError("No valid prediction arrays found in all_test_preds.")

    # Stack predictions into a 2D array: (n_models_or_folds, n_samples)
    # This will raise a ValueError if shapes are inconsistent, which is desired (propagate error).
    stacked_preds = np.stack(test_preds_to_average)

    # Calculate the simple mean across the model/fold axis (axis=0)
    # This implements the equal weighting (1/N) strategy.
    final_test_preds = np.mean(stacked_preds, axis=0)

    # Step 3: Return final test predictions
    # Requirement: Output must NOT contain NaN or Infinity values.
    if not np.isfinite(final_test_preds).all():
        raise RuntimeError("Ensemble result contains non-finite values (NaN or Inf).")

    return final_test_preds
