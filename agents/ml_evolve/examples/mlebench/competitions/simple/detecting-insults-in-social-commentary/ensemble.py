# -*- coding: utf-8 -*-
"""
LLM Generated Code
"""

from typing import Dict, List

import numpy as np
import pandas as pd

BASE_DATA_PATH = "/root/workspace/evolux/output/mlebench/detecting-insults-in-social-commentary/prepared/public"
OUTPUT_DATA_PATH = "output/bb8b0571-64fb-49b6-8e49-1fc2d52da49b/2/executor/output"

# Type Hints
DT = pd.DataFrame | pd.Series | np.ndarray


def ensemble(
    all_oof_preds: Dict[str, DT],
    all_test_preds: Dict[str, List[DT]],
    y_true_full: DT
) -> DT:
    """
    Ensembles predictions from multiple models by averaging fold-wise predictions
    and then across all models.

    Args:
        all_oof_preds: Dictionary of {model_name: oof_predictions}.
        all_test_preds: Dictionary of {model_name: [pred_fold_1, pred_fold_2, ...]}.
        y_true_full: The Ground Truth labels.

    Returns:
        DT: Final aggregated predictions for the Test set.
    """
    # Step 1: Check if there are any predictions to ensemble
    if not all_test_preds:
        raise ValueError("The 'all_test_preds' dictionary is empty. No predictions to ensemble.")

    model_test_averages = []

    # Step 2: Aggregate fold predictions for each model (Collapse List[DT] -> DT)
    for model_name, fold_preds_list in all_test_preds.items():
        if not fold_preds_list:
            continue

        # Convert list of predictions (can be Series or ndarray) into a 2D numpy array
        # Shape: (num_folds, num_test_samples)
        fold_preds_array = np.array([np.asarray(p) for p in fold_preds_list])

        # Compute the average prediction across all folds for this model
        model_avg = np.mean(fold_preds_array, axis=0)
        model_test_averages.append(model_avg)

    if not model_test_averages:
        raise ValueError("No valid fold predictions were found to aggregate.")

    # Step 3: Apply ensemble strategy - Simple average across all registered models
    # If there is only one model (like Logistic Regression), this returns the fold-averaged predictions.
    final_test_preds = np.mean(model_test_averages, axis=0)

    # Step 4: Final checks for data integrity
    # Ensure output does not contain NaN or Infinity values
    if np.any(np.isnan(final_test_preds)):
        raise ValueError("Ensemble final predictions contain NaN values.")
    if np.any(np.isinf(final_test_preds)):
        raise ValueError("Ensemble final predictions contain Infinity values.")

    # Return final predictions
    return final_test_preds
