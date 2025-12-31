# -*- coding: utf-8 -*-
"""
LLM Generated Code
"""

from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import rankdata

BASE_DATA_PATH = "/root/workspace/evolux_ml/output/mlebench/google-quest-challenge/prepared/public"
OUTPUT_DATA_PATH = "output/ea4f9f02-c6be-4ad0-a6fd-71930e8fb81e/15/executor/output"

# Type Hints
DT = pd.DataFrame | pd.Series | np.ndarray


def ensemble(
    all_oof_preds: Dict[str, DT],
    all_test_preds: Dict[str, List[DT]],
    y_true_full: DT
) -> DT:
    """
    Ensembles predictions from multiple models using a Multi-stage Rank Averaging strategy.

    Args:
        all_oof_preds: Dictionary of {model_name: oof_predictions}.
        all_test_preds: Dictionary of {model_name: [pred_fold_1, pred_fold_2, ...]}.
        y_true_full: The Ground Truth labels (used for column names and alignment).

    Returns:
        DT: Final ensembled predictions for the Test set.
    """
    # Step 1: Fold Averaging
    # For each model, compute the arithmetic mean of its fold predictions to stabilize the signal.
    averaged_model_preds = {}
    for model_name, folds in all_test_preds.items():
        if not folds:
            continue

        # Convert all folds to numpy arrays for mean calculation
        fold_arrays = []
        for f in folds:
            if isinstance(f, (pd.DataFrame, pd.Series)):
                fold_arrays.append(f.values)
            else:
                fold_arrays.append(np.array(f))

        # Compute arithmetic mean across folds for this specific model architecture
        averaged_model_preds[model_name] = np.mean(fold_arrays, axis=0)

    if not averaged_model_preds:
        raise ValueError("No predictions found in all_test_preds to ensemble.")

    # Step 2: Model Fusion via Rank Averaging
    # Consolidates different architectures (e.g., DeBERTa and RoBERTa) using ranks.
    model_names = list(averaged_model_preds.keys())
    # Determine dimensions from the first available model
    n_samples, n_targets = averaged_model_preds[model_names[0]].shape
    final_test_preds = np.zeros((n_samples, n_targets))

    for col_idx in range(n_targets):
        # Accumulate ranks for the current target column across all models
        combined_ranks = np.zeros(n_samples)
        for model_name in model_names:
            pred_col = averaged_model_preds[model_name][:, col_idx]
            # rankdata handles ties by averaging ranks, which aligns with Spearman correlation behavior
            combined_ranks += rankdata(pred_col)

        # Normalize the accumulated ranks to the [0, 1] range for the specific target
        rank_min = combined_ranks.min()
        rank_max = combined_ranks.max()

        if rank_max > rank_min:
            final_test_preds[:, col_idx] = (combined_ranks - rank_min) / (rank_max - rank_min)
        else:
            # Fallback for constant predictions: use the arithmetic mean of the original values
            # This ensures no NaNs are produced if all models predict a constant for a column.
            final_test_preds[:, col_idx] = np.mean([averaged_model_preds[m][:, col_idx] for m in model_names], axis=0)

    # Step 3: Final clipping to ensure all values are strictly within [0, 1]
    final_test_preds = np.clip(final_test_preds, 0.0, 1.0)

    # Ensure output maintains target column names if y_true_full is a DataFrame
    if isinstance(y_true_full, pd.DataFrame):
        return pd.DataFrame(final_test_preds, columns=y_true_full.columns)

    return final_test_preds
