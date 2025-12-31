# -*- coding: utf-8 -*-
"""
LLM Generated Code
"""

from typing import Dict, List

import numpy as np
import pandas as pd

BASE_DATA_PATH = "/root/workspace/evolux/output/mlebench/tabular-playground-series-dec-2021/prepared/public"
OUTPUT_DATA_PATH = "output/ead961ce-50a1-41ec-89e9-4cc0d527fbe5/1/executor/output"

# Type Hints
DT = pd.DataFrame | pd.Series | np.ndarray


def ensemble(
    all_oof_preds: Dict[str, DT],
    all_test_preds: Dict[str, List[DT]],
    y_true_full: DT
) -> DT:
    """
    Ensembles predictions from multiple models using soft voting.

    Args:
        all_oof_preds: Dictionary of {model_name: oof_predictions}.
        all_test_preds: Dictionary of {model_name: [pred_fold_1, pred_fold_2, ...]}.
        y_true_full: The Ground Truth labels (use for optimization/scoring).

    Returns:
        DT: Final predictions for the Test set.
    """
    # Step 1: Aggregate fold predictions for each model
    # We collect all probability matrices from all folds of all models into a single list.
    # This allows us to perform a global soft voting across all available predictions.
    all_matrices = []
    for model_name, fold_preds in all_test_preds.items():
        for fold_pred in fold_preds:
            # Convert to numpy array if it's a pandas object to ensure numeric operations are consistent
            if isinstance(fold_pred, (pd.DataFrame, pd.Series)):
                all_matrices.append(fold_pred.values)
            else:
                all_matrices.append(fold_pred)

    if not all_matrices:
        raise ValueError("The input all_test_preds is empty or contains no valid predictions.")

    # Step 2: Apply ensemble strategy (Soft Voting)
    # Calculate the mean probability for each class across all folds and models.
    # Expected shape of mean_probs: (n_samples, n_classes)
    mean_probs = np.mean(all_matrices, axis=0)

    # Safety check for prediction validity
    if np.isnan(mean_probs).any() or np.isinf(mean_probs).any():
        raise ValueError("The averaged probabilities contain NaN or Infinity values.")

    # Step 3: Final Prediction using argmax
    # Get the index of the class with the highest average probability.
    # Expected shape of final_indices: (n_samples,)
    final_indices = np.argmax(mean_probs, axis=1)

    # Class Mapping: Map internal model labels (0-5) back to original Cover_Type (1, 2, 3, 4, 6, 7).
    # The load_data component used the following mapping:
    # {1: 0, 2: 1, 3: 2, 4: 3, 6: 4, 7: 5}
    # Therefore, our inverse mapping is defined as:
    inverse_map = np.array([1, 2, 3, 4, 6, 7])

    # Use vectorized indexing to map internal class indices to original labels
    final_test_predictions = inverse_map[final_indices]

    # Return final test predictions as a numpy array
    return final_test_predictions
