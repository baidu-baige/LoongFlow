# -*- coding: utf-8 -*-
"""
LLM Generated Code
"""

from typing import Any, Dict, List

import numpy as np
import pandas as pd

BASE_DATA_PATH = "/root/workspace/evolux/output/mlebench/mlsp-2013-birds/prepared/public"
OUTPUT_DATA_PATH = "output/b4d4dfce-4367-41a9-8cac-a59279a6d65f/11/executor/output"

# Type Hints
DT = pd.DataFrame | pd.Series | np.ndarray


def ensemble(
    all_oof_preds: Dict[str, DT],
    all_test_preds: Dict[str, List[DT]],
    y_true_full: DT
) -> DT:
    """
    Ensembles predictions from multiple models using Strict Simple Averaging.

    Strategy:
    1. Aggregates fold predictions for each model (Bagging) using mean.
    2. Computes the grand average across all models (Simple Average).
    3. No weighting or optimization logic is applied to prevent overfitting on small/sparse data.

    Args:
        all_oof_preds: Dictionary of {model_name: oof_predictions}. (Unused for simple average)
        all_test_preds: Dictionary of {model_name: [pred_fold_1, pred_fold_2, ...]}.
        y_true_full: The Ground Truth labels. (Unused for simple average)

    Returns:
        DT: Final predictions for the Test set.
    """

    # Helper to ensure data is in numpy format
    def to_numpy(data: Any) -> np.ndarray:
        if isinstance(data, pd.DataFrame):
            return data.values
        if isinstance(data, pd.Series):
            return data.values
        if isinstance(data, np.ndarray):
            return data
        if isinstance(data, list):
            return np.array(data)
        return np.array(data)

    model_averaged_preds = []

    # Step 1: Aggregate fold predictions for each model (Bagging)
    for model_name, fold_preds_list in all_test_preds.items():
        if not fold_preds_list:
            continue

        # Convert all fold predictions to numpy arrays
        try:
            fold_preds_np = [to_numpy(pred) for pred in fold_preds_list]
        except Exception as e:
            print(f"Ensemble Warning: Error converting predictions for model '{model_name}': {e}")
            continue

        if not fold_preds_np:
            continue

        # Check shape consistency across folds
        ref_shape = fold_preds_np[0].shape
        valid_folds = []
        for p in fold_preds_np:
            if p.shape == ref_shape:
                valid_folds.append(p)
            else:
                print(
                    f"Ensemble Warning: Shape mismatch in folds for model '{model_name}'. Got {p.shape}, expected {ref_shape}. Skipping fold.")

        if not valid_folds:
            continue

        # Stack arrays along a new axis to compute stats across folds
        # Shape: (n_folds, n_samples, n_classes)
        stacked_folds = np.array(valid_folds)

        # Compute the mean prediction across all folds for this model
        # Shape: (n_samples, n_classes)
        model_mean = np.mean(stacked_folds, axis=0)

        model_averaged_preds.append(model_mean)

    if not model_averaged_preds:
        raise ValueError("No valid predictions provided in all_test_preds to ensemble.")

    # Step 2: Choose Strategy (Strict Simple Average)
    # Shape: (n_models, n_samples, n_classes)
    meta_stack = np.array(model_averaged_preds)

    # Step 3: Compute final ensemble
    # Shape: (n_samples, n_classes)
    final_predictions = np.mean(meta_stack, axis=0)

    # Ensure stability by replacing NaNs and Infinities
    final_predictions = np.nan_to_num(final_predictions, nan=0.0, posinf=1.0, neginf=0.0)

    # Clip probabilities to [0, 1] range
    final_predictions = np.clip(final_predictions, 0.0, 1.0)

    return final_predictions
