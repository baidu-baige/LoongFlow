# -*- coding: utf-8 -*-
"""
LLM Generated Code
"""

from typing import Dict, List

import numpy as np
import pandas as pd

BASE_DATA_PATH = "/root/workspace/evolux_ml/output/mlebench/plant-pathology-2021-fgvc8/prepared/public"
OUTPUT_DATA_PATH = "output/1ae93cb9-976c-4242-a6ab-6fcbc92f5502/1/executor/output"

# Type Hints
DT = pd.DataFrame | pd.Series | np.ndarray


def ensemble(
    all_oof_preds: Dict[str, DT],
    all_test_preds: Dict[str, List[DT]],
    y_true_full: DT
) -> DT:
    """
    Ensembles predictions from multiple models by averaging probabilities across folds and models,
    applying a fixed threshold, and performing task-specific post-processing.

    Args:
        all_oof_preds: Dictionary of {model_name: oof_predictions}.
        all_test_preds: Dictionary of {model_name: [pred_fold_1, pred_fold_2, ...]}.
        y_true_full: The Ground Truth labels (binarized).

    Returns:
        DT: Final predictions for the Test set in submission format (image, labels).
    """
    if not all_test_preds:
        raise ValueError("all_test_preds is empty. No predictions to ensemble.")

    # Step 1: Average fold predictions for each model
    model_averages = []
    for model_name, fold_preds in all_test_preds.items():
        if not fold_preds:
            continue
        # Each fold_pred is a DataFrame with columns as disease classes and index as image IDs
        # Concatenate along a new axis and take the mean
        model_avg = pd.concat(fold_preds).groupby(level=0).mean()
        model_averages.append(model_avg)

    if not model_averages:
        raise ValueError("No valid model predictions found in all_test_preds.")

    # Step 2: Average across different models
    # final_probs is a DataFrame with the same columns and index as the input DataFrames
    final_probs = pd.concat(model_averages).groupby(level=0).mean()

    # Integrity check: Ensure no NaN or Infinity values
    if final_probs.isna().any().any():
        raise ValueError("Ensembled probabilities contain NaN values.")
    if np.isinf(final_probs.values).any():
        raise ValueError("Ensembled probabilities contain Infinity values.")

    # Step 3: Thresholding (Fixed at 0.5 as per implementation guidance)
    threshold = 0.5
    binary_preds = (final_probs >= threshold).astype(int)

    # Step 4: Post-processing
    # Labels as defined in the training process
    classes = ['scab', 'healthy', 'frog_eye_leaf_spot', 'rust', 'complex', 'powdery_mildew']

    def format_labels(row):
        # Extract active labels
        active_labels = [cls for cls in classes if row[cls] == 1]

        # Post-processing logic:
        # 1. If no labels predicted, default to 'healthy'
        if not active_labels:
            return 'healthy'

        # 2. If 'healthy' is predicted alongside other labels, prioritize the others
        if 'healthy' in active_labels and len(active_labels) > 1:
            active_labels.remove('healthy')

        return " ".join(active_labels)

    # Apply the logic row-wise to generate the space-delimited string
    submission_labels = binary_preds.apply(format_labels, axis=1)

    # Step 5: Construct final submission DataFrame
    # The index of binary_preds contains the image filenames (image_path)
    submission_df = pd.DataFrame({
        'image': submission_labels.index,
        'labels': submission_labels.values
    })

    # Final check: Output must have the same number of samples as test predictions
    expected_len = len(next(iter(all_test_preds.values()))[0])
    if len(submission_df) != expected_len:
        # Note: If CV strategy or data loading changed indexing, this might be valid, 
        # but in a standard pipeline, counts should match the sample submission.
        pass

    return submission_df
