# -*- coding: utf-8 -*-
"""
LLM Generated Code
"""

from typing import Dict, List

import numpy as np
import pandas as pd

BASE_DATA_PATH = "/root/workspace/evolux/output/mlebench/plant-pathology-2020-fgvc7/prepared/public"
OUTPUT_DATA_PATH = "output/1799683e-18f2-43a3-97d1-8b0bdddc3200/1/executor/output"

# Type Hints
DT = pd.DataFrame | pd.Series | np.ndarray


def ensemble(
    all_oof_preds: Dict[str, DT],
    all_test_preds: Dict[str, List[DT]],
    y_true_full: DT
) -> DT:
    """
    Ensembles predictions from multiple models by averaging fold predictions.

    Args:
        all_oof_preds: Dictionary of {model_name: oof_predictions}.
        all_test_preds: Dictionary of {model_name: [pred_fold_1, pred_fold_2, ...]}.
        y_true_full: The Ground Truth labels (use for optimization/scoring).

    Returns:
        DT: Final predictions for the Test set.
    """
    # Step 1: Aggregate fold predictions for each model
    # We gather every prediction array provided across all models and all folds.
    # The implementation guidance specifies averaging the predictions from all 5 models
    # (one from each fold).
    all_fold_predictions = []
    for model_name, fold_list in all_test_preds.items():
        for fold_pred in fold_list:
            # Convert to numpy array to ensure consistency (handles DataFrames/Series)
            all_fold_predictions.append(np.asarray(fold_pred))

    # Step 2: Apply ensemble strategy
    # Calculate the mean across all collected prediction arrays.
    # axis=0 computes the mean for each sample across all folds/models.
    # This effectively performs the "Fold Averaging" requirement.
    # (Note: If TTA predictions were included in the input lists, they are averaged here too).
    final_test_preds = np.mean(all_fold_predictions, axis=0)

    # Step 3: Ensure the final output is a probability distribution (sum to 1)
    # for each image across the 4 classes.
    # While the average of valid probability distributions (softmax outputs)
    # is itself a probability distribution, we normalize explicitly for robustness.
    row_sums = final_test_preds.sum(axis=1, keepdims=True)
    final_test_preds = final_test_preds / row_sums

    # Check for NaN/Inf (requirements specify propagation of errors,
    # but the logic itself should not introduce them).
    if np.isnan(final_test_preds).any() or np.isinf(final_test_preds).any():
        # Let the error propagate as per requirements if inputs caused invalid values
        pass

    return final_test_preds
