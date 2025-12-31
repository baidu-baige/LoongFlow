# -*- coding: utf-8 -*-
"""
LLM Generated Code
"""

from typing import Dict, List

import numpy as np
import pandas as pd

BASE_DATA_PATH = "/root/workspace/evolux/output/mlebench/jigsaw-toxic-comment-classification-challenge/prepared/public"
OUTPUT_DATA_PATH = "output/40d01db3-cd9d-46e2-8d9c-d192fb8addff/1/executor/output"

# Type Hints
DT = pd.DataFrame | pd.Series | np.ndarray


def ensemble(
    all_oof_preds: Dict[str, DT],
    all_test_preds: Dict[str, List[DT]],
    y_true_full: DT
) -> DT:
    """
    Ensembles predictions from multiple models using simple arithmetic mean.

    Args:
        all_oof_preds: Dictionary of {model_name: oof_predictions}.
        all_test_preds: Dictionary of {model_name: [pred_fold_1, pred_fold_2, ...]}.
        y_true_full: The Ground Truth labels.

    Returns:
        DT: Final averaged predictions for the Test set.
    """
    # Step 1: Collect all fold predictions from all models
    all_preds_to_average = []

    for model_name, fold_preds_list in all_test_preds.items():
        for pred in fold_preds_list:
            # Ensure each prediction is a DataFrame to facilitate alignment and averaging
            if isinstance(pred, np.ndarray):
                # If it's a numpy array, we convert to DataFrame. 
                # We assume the column order matches the target columns from y_true_full.
                cols = y_true_full.columns if hasattr(y_true_full, 'columns') else None
                pred_df = pd.DataFrame(pred, columns=cols)
                all_preds_to_average.append(pred_df)
            else:
                all_preds_to_average.append(pred)

    if not all_preds_to_average:
        raise ValueError("The all_test_preds dictionary is empty or contains no predictions.")

    # Step 2: Apply ensemble strategy (Simple Arithmetic Mean)
    # Concatenate all DataFrames and group by index to calculate mean
    # This automatically handles multiple models and multiple folds per model.
    ensemble_df = pd.concat(all_preds_to_average).groupby(level=0).mean()

    # Step 3: Verification
    # Ensure no NaNs or Infs are present in the final output
    if ensemble_df.isnull().values.any():
        raise ValueError("Ensemble output contains NaN values. Check input predictions for missing data.")

    if np.isinf(ensemble_df.values).any():
        raise ValueError(
            "Ensemble output contains Infinity values. Check input predictions for numerical stability issues.")

    # Ensure the columns match the required submission columns (order and names)
    if hasattr(y_true_full, 'columns'):
        target_cols = list(y_true_full.columns)
        # Reorder columns just in case, while propagating errors if columns are missing
        ensemble_df = ensemble_df[target_cols]

    return ensemble_df
