# -*- coding: utf-8 -*-
"""
LLM Generated Code
"""

from typing import Dict, List

import numpy as np
import pandas as pd

BASE_DATA_PATH = "/root/workspace/evolux/output/mlebench/cassava-leaf-disease-classification/prepared/public"
OUTPUT_DATA_PATH = "output/068f0c14-e630-462e-bc46-9d2d4b1d5fc3/5/executor/output"

# Type Hints
DT = pd.DataFrame | pd.Series | np.ndarray


def ensemble(
    all_oof_preds: Dict[str, DT],
    all_test_preds: Dict[str, List[DT]],
    y_true_full: DT
) -> DT:
    """
    Ensembles predictions from multiple models using soft-voting across folds.
    This implementation assumes that the input predictions (all_test_preds) already
    incorporate multi-view TTA (Test-Time Augmentation) as defined in the training stage.

    Args:
        all_oof_preds: Dictionary of {model_name: oof_predictions}.
        all_test_preds: Dictionary of {model_name: [pred_fold_1, pred_fold_2, ...]}.
        y_true_full: The Ground Truth labels (used for optimization/scoring).

    Returns:
        DT: Final predictions for the Test set (class indices).
    """
    total_test_probs = None
    num_prediction_sets = 0

    # Step 1: Iterate through each model and its corresponding list of fold predictions
    for model_name, fold_preds_list in all_test_preds.items():
        for fold_probs in fold_preds_list:
            # Ensure fold_probs is a numpy array for vectorization
            # Each fold_probs is expected to be an (N, 5) array of softmax probabilities
            # containing the average of 4-view TTA: (Orig + HF + VF + Transpose) / 4
            probs_array = np.array(fold_probs, dtype=np.float64)

            if total_test_probs is None:
                total_test_probs = np.zeros_like(probs_array, dtype=np.float64)

            # Step 2: Accumulate probabilities for soft voting
            total_test_probs += probs_array
            num_prediction_sets += 1

    if num_prediction_sets == 0:
        raise ValueError("The all_test_preds dictionary is empty or contains no fold predictions.")

    # Step 3: Compute the average probability distribution across all models and folds
    # Logic: sum(fold_probs) / (num_models * num_folds)
    avg_test_probs = total_test_probs / num_prediction_sets

    # Step 4: Validate output for quality - ensure no NaN or Infinity values
    if np.isnan(avg_test_probs).any() or np.isinf(avg_test_probs).any():
        raise ValueError("Ensembled probabilities contain NaN or Infinity values.")

    # Step 5: Final Decision - Apply np.argmax to select the class with the highest averaged probability
    final_test_predictions = np.argmax(avg_test_probs, axis=1)

    return final_test_predictions
