# -*- coding: utf-8 -*-
"""
LLM Generated Code
"""

from typing import Dict, List

import numpy as np
import pandas as pd

BASE_DATA_PATH = "/root/workspace/evolux/output/mlebench/aerial-cactus-identification/prepared/public"
OUTPUT_DATA_PATH = "output/153e4624-940b-4d19-a37d-90435531bfd1/1/executor/output"

# Type Hints
DT = pd.DataFrame | pd.Series | np.ndarray


def ensemble(
    all_oof_preds: Dict[str, DT],
    all_test_preds: Dict[str, List[DT]],
    y_true_full: DT
) -> DT:
    """
    Ensembles predictions from multiple models.
    
    For this first iteration, we focus on establishing a strong single-model baseline.
    The strategy is:
    - If using K-fold CV, average the predictions from all folds
    - For a single model, simply aggregate fold predictions
    - Store out-of-fold predictions for potential future stacking
    
    Args:
        all_oof_preds: Dictionary of {model_name: oof_predictions}.
        all_test_preds: Dictionary of {model_name: [pred_fold_1, pred_fold_2, ...]}.
        y_true_full: The Ground Truth labels (use for optimization/scoring).

    Returns:
        DT: Final predictions for the Test set.
    """
    from sklearn.metrics import roc_auc_score

    # Step 1: Aggregate fold predictions for each model
    # For each model, average the test predictions across all folds
    aggregated_test_preds = {}

    for model_name, fold_preds_list in all_test_preds.items():
        if len(fold_preds_list) == 0:
            continue

        # Convert each fold prediction to numpy array
        fold_arrays = []
        for fold_pred in fold_preds_list:
            if isinstance(fold_pred, (pd.DataFrame, pd.Series)):
                fold_arrays.append(fold_pred.values.flatten())
            else:
                fold_arrays.append(np.array(fold_pred).flatten())

        # Stack and average across folds
        stacked_preds = np.stack(fold_arrays, axis=0)
        avg_pred = np.mean(stacked_preds, axis=0)
        aggregated_test_preds[model_name] = avg_pred

    # Step 2: Calculate OOF scores for each model (for logging/future weighting)
    oof_scores = {}
    for model_name, oof_pred in all_oof_preds.items():
        if isinstance(oof_pred, (pd.DataFrame, pd.Series)):
            oof_array = oof_pred.values.flatten()
        else:
            oof_array = np.array(oof_pred).flatten()

        if isinstance(y_true_full, (pd.DataFrame, pd.Series)):
            y_true_array = y_true_full.values.flatten()
        else:
            y_true_array = np.array(y_true_full).flatten()

        # Calculate AUC-ROC score
        try:
            score = roc_auc_score(y_true_array, oof_array)
            oof_scores[model_name] = score
            print(f"Model '{model_name}' OOF AUC-ROC: {score:.6f}")
        except Exception as e:
            print(f"Could not calculate OOF score for '{model_name}': {e}")
            oof_scores[model_name] = 0.5  # Default score

    # Step 3: Compute final ensemble
    # For this first iteration with a single model, we simply use the aggregated predictions
    # Future iterations can implement weighted averaging based on OOF scores

    if len(aggregated_test_preds) == 0:
        raise ValueError("No valid test predictions found to ensemble")

    if len(aggregated_test_preds) == 1:
        # Single model case - just return its predictions
        model_name = list(aggregated_test_preds.keys())[0]
        final_predictions = aggregated_test_preds[model_name]
        print(f"Single model ensemble using '{model_name}'")
    else:
        # Multiple models - use simple averaging for now
        # Future iterations can use weighted averaging based on OOF scores
        print(f"Ensembling {len(aggregated_test_preds)} models using simple averaging")

        # Stack all model predictions
        all_model_preds = []
        for model_name, preds in aggregated_test_preds.items():
            all_model_preds.append(preds)

        stacked_model_preds = np.stack(all_model_preds, axis=0)
        final_predictions = np.mean(stacked_model_preds, axis=0)

    # Step 4: Post-processing - ensure valid probability range
    final_predictions = np.clip(final_predictions, 1e-7, 1 - 1e-7)

    # Validate output
    assert not np.isnan(final_predictions).any(), "Final predictions contain NaN values"
    assert not np.isinf(final_predictions).any(), "Final predictions contain Inf values"

    # Get expected length from first model's first fold
    first_model = list(all_test_preds.keys())[0]
    first_fold_pred = all_test_preds[first_model][0]
    if isinstance(first_fold_pred, (pd.DataFrame, pd.Series)):
        expected_length = len(first_fold_pred)
    else:
        expected_length = len(first_fold_pred)

    assert len(final_predictions) == expected_length, \
        f"Output length {len(final_predictions)} does not match expected {expected_length}"

    print(f"Final ensemble predictions shape: {final_predictions.shape}")
    print(f"Prediction range: [{final_predictions.min():.6f}, {final_predictions.max():.6f}]")

    return final_predictions
