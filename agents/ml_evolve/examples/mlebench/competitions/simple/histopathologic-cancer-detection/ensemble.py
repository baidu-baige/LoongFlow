# -*- coding: utf-8 -*-
"""
LLM Generated Code
"""

import os
from typing import Dict, List

import numpy as np
import pandas as pd

BASE_DATA_PATH = "/root/workspace/evolux_ml/output/mlebench/histopathologic-cancer-detection/prepared/public"
OUTPUT_DATA_PATH = "output/ee5d67f1-b3aa-447c-b45e-155ce5c4f09c/1/executor/output"

# Type Hints
DT = pd.DataFrame | pd.Series | np.ndarray


def ensemble(
    all_oof_preds: Dict[str, DT],
    all_test_preds: Dict[str, List[DT]],
    y_true_full: DT
) -> DT:
    """
    Ensembles predictions from multiple models using arithmetic averaging.

    Args:
        all_oof_preds: Dictionary of {model_name: oof_predictions}.
        all_test_preds: Dictionary of {model_name: [pred_fold_1, pred_fold_2, ...]}.
        y_true_full: The Ground Truth labels (not used for simple averaging).

    Returns:
        DT: Final predictions for the Test set in sample_submission format.
    """
    # Step 1: Aggregate all fold predictions from all models into a single list
    # We convert each prediction to a numpy array to ensure consistent behavior
    all_fold_predictions = []
    for model_name, fold_list in all_test_preds.items():
        for fold_pred in fold_list:
            all_fold_predictions.append(np.asarray(fold_pred))

    if not all_fold_predictions:
        raise ValueError("The all_test_preds dictionary is empty or contains no predictions.")

    # Step 2: Apply ensemble strategy (Arithmetic Averaging)
    # Calculate the mean probability across all available folds and models
    # np.mean with axis=0 will average across the list of 1D arrays
    final_test_probs = np.mean(all_fold_predictions, axis=0)

    # Robustness check: Ensure results do not contain invalid numerical values
    if np.any(np.isnan(final_test_probs)) or np.any(np.isinf(final_test_probs)):
        raise ValueError("Ensemble calculation resulted in NaN or Infinity values.")

    # Step 3: Format the output to match sample_submission.csv (id, label)
    # Load the original sample submission to retrieve the correct IDs
    sample_sub_path = os.path.join(BASE_DATA_PATH, "sample_submission.csv")
    sample_sub = pd.read_csv(sample_sub_path)

    # If the predictions are from a subset (e.g., validation_mode = True),
    # we truncate the sample_sub to match the length of the predictions.
    # This assumes the predictions follow the same sequential order as the original file.
    if len(final_test_probs) < len(sample_sub):
        sample_sub = sample_sub.iloc[:len(final_test_probs)].copy()
    elif len(final_test_probs) > len(sample_sub):
        raise ValueError(
            f"Received {len(final_test_probs)} predictions, but sample submission only contains {len(sample_sub)} IDs.")

    # Construct the final submission DataFrame
    submission_df = pd.DataFrame({
        'id': sample_sub['id'].values,
        'label': final_test_probs
    })

    return submission_df
