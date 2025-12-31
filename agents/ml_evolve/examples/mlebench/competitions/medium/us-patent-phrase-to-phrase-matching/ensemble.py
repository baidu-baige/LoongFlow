# -*- coding: utf-8 -*-
"""
LLM Generated Code
"""

from typing import Dict, List

import numpy as np
import pandas as pd

BASE_DATA_PATH = "/root/workspace/evolux/output/mlebench/us-patent-phrase-to-phrase-matching/prepared/public"
OUTPUT_DATA_PATH = "output/aefcc010-8f21-4ecb-b149-7bf99579e6d3/6/executor/output"

# Type Hints
DT = pd.DataFrame | pd.Series | np.ndarray


def ensemble(
    all_oof_preds: Dict[str, DT],
    all_test_preds: Dict[str, List[DT]],
    y_true_full: DT
) -> DT:
    """
    Ensembles predictions from multiple models using arithmetic averaging across folds.

    Args:
        all_oof_preds: Dictionary of {model_name: oof_predictions}.
        all_test_preds: Dictionary of {model_name: [pred_fold_1, pred_fold_2, ...]}.
        y_true_full: The Ground Truth labels (use for optimization/scoring).

    Returns:
        DT: Final predictions for the Test set.
    """
    # Step 1: Collect all individual fold predictions from all models
    # We iterate through each model and each fold's test predictions.
    fold_predictions = []
    for model_name, preds_list in all_test_preds.items():
        for p in preds_list:
            # Ensure the prediction is converted to a numpy array for consistent math
            fold_predictions.append(np.asanyarray(p))

    # Step 2: Apply ensemble strategy - Arithmetic Averaging
    # We calculate the mean across all collected fold predictions.
    # axis=0 averages element-wise across the samples.
    final_predictions = np.mean(fold_predictions, axis=0)

    # Step 3: Apply constraints and post-processing
    # Clip results to the competition range [0.0, 1.0].
    # This also handles positive/negative infinity by mapping them to 1.0/0.0 respectively.
    final_predictions = np.clip(final_predictions, 0.0, 1.0)

    # Fill any potential NaN values with the global training mean (0.362).
    # This ensures the output is robust and contains no invalid values.
    final_predictions = np.where(np.isnan(final_predictions), 0.362, final_predictions)

    # Final verification for Infinity (though handled by clip, we ensure strict compliance)
    if not np.isfinite(final_predictions).all():
        # If non-finite values still exist (e.g. NaN logic failure), we replace them.
        final_predictions[~np.isfinite(final_predictions)] = 0.362

    # The output order is maintained as long as the inputs follow the test set order.
    return final_predictions
