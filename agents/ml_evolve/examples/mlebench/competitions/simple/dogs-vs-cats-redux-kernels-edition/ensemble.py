# -*- coding: utf-8 -*-
"""
LLM Generated Code
"""

from typing import Dict, List

import numpy as np
import pandas as pd

BASE_DATA_PATH = "/root/workspace/evolux/output/mlebench/dogs-vs-cats-redux-kernels-edition/prepared/public"
OUTPUT_DATA_PATH = "output/7dbf3696-f36d-4f87-8e59-bb45d92869c5/1/executor/output"

# Type Hints
DT = pd.DataFrame | pd.Series | np.ndarray


def ensemble(
    all_oof_preds: Dict[str, DT],
    all_test_preds: Dict[str, List[DT]],
    y_true_full: DT
) -> DT:
    """
    Ensembles predictions from multiple models using a simple arithmetic mean.

    Args:
        all_oof_preds: Dictionary of {model_name: oof_predictions}.
        all_test_preds: Dictionary of {model_name: [pred_fold_1, pred_fold_2, ...]}.
        y_true_full: The Ground Truth labels.

    Returns:
        DT: Final aggregated predictions for the Test set.
    """
    # Step 1: Aggregate fold predictions for each model (Collapse List[DT] -> DT)
    # We calculate the mean of all folds for each model separately, then average the models.
    # This ensures each model contributes equally to the ensemble regardless of its fold count.

    model_averages = []

    for model_name, fold_preds in all_test_preds.items():
        if not fold_preds:
            continue

        # Convert each fold's predictions to a flat numpy array
        # fold_preds is a list of DT (pd.Series or np.ndarray)
        fold_arrays = [np.asarray(p).flatten() for p in fold_preds]

        # Compute the mean across all folds for this model
        # Resulting shape: (num_test_samples,)
        model_mean = np.mean(np.stack(fold_arrays), axis=0)
        model_averages.append(model_mean)

    if not model_averages:
        raise ValueError("The provided all_test_preds dictionary is empty or contains no predictions.")

    # Step 2: Apply ensemble strategy (Arithmetic Mean across models)
    # Combine the averaged predictions from different models
    final_test_preds = np.mean(np.stack(model_averages), axis=0)

    # Step 3: Validate and Return final test predictions
    # Ensure there are no NaNs or Infs that could lead to invalid submissions
    if np.isnan(final_test_preds).any() or np.isinf(final_test_preds).any():
        raise ValueError("Ensemble process generated NaN or Infinity values in the final predictions.")

    return final_test_preds
