# -*- coding: utf-8 -*-
"""
LLM Generated Code
"""

import os
from typing import Dict, List

import numpy as np
import pandas as pd

BASE_DATA_PATH = "/root/workspace/evolux/output/mlebench/nomad2018-predict-transparent-conductors/prepared/public"
OUTPUT_DATA_PATH = "output/2ea9a3e6-0185-40d8-bd93-7afe244a50a1/2/executor/output"

# Type Hints
DT = pd.DataFrame | pd.Series | np.ndarray


def ensemble(
    all_oof_preds: Dict[str, DT],
    all_test_preds: Dict[str, List[DT]],
    y_true_full: DT
) -> DT:
    """
    Ensembles predictions from multiple models.

    For this initial iteration, we focus on establishing a strong baseline
    using simple averaging of available models (LightGBM and XGBoost).
    
    The targets have been log1p transformed in create_features, so we need
    to apply expm1 to convert predictions back to original scale.

    Args:
        all_oof_preds: Dictionary of {model_name: oof_predictions}.
        all_test_preds: Dictionary of {model_name: [pred_fold_1, pred_fold_2, ...]}.
        y_true_full: The Ground Truth labels (use for optimization/scoring).

    Returns:
        DT: Final predictions for the Test set.
    """

    # Helper function to compute RMSLE
    def compute_rmsle(y_true, y_pred):
        """Compute Root Mean Squared Logarithmic Error."""
        # Ensure non-negative values
        y_pred_clipped = np.maximum(y_pred, 0)
        y_true_clipped = np.maximum(y_true, 0)

        log_pred = np.log1p(y_pred_clipped)
        log_true = np.log1p(y_true_clipped)

        return np.sqrt(np.mean((log_pred - log_true) ** 2))

    # Step 1: Aggregate fold predictions for each model
    # Average predictions across folds for each model
    aggregated_test_preds = {}

    for model_name, fold_preds_list in all_test_preds.items():
        if len(fold_preds_list) == 0:
            continue

        # Stack all fold predictions and compute mean
        stacked_preds = []
        for fold_pred in fold_preds_list:
            if isinstance(fold_pred, pd.DataFrame):
                stacked_preds.append(fold_pred.values)
            elif isinstance(fold_pred, pd.Series):
                stacked_preds.append(fold_pred.values.reshape(-1, 1))
            else:
                arr = np.array(fold_pred)
                if arr.ndim == 1:
                    arr = arr.reshape(-1, 1)
                stacked_preds.append(arr)

        # Average across folds
        stacked_array = np.stack(stacked_preds, axis=0)
        mean_pred = np.mean(stacked_array, axis=0)
        aggregated_test_preds[model_name] = mean_pred

    # Step 2: Compute OOF scores for each model (for logging/future use)
    # Convert y_true_full to numpy array
    if isinstance(y_true_full, pd.DataFrame):
        y_true_np = y_true_full.values
        target_columns = y_true_full.columns.tolist()
    elif isinstance(y_true_full, pd.Series):
        y_true_np = y_true_full.values.reshape(-1, 1)
        target_columns = ['target']
    else:
        y_true_np = np.array(y_true_full)
        if y_true_np.ndim == 1:
            y_true_np = y_true_np.reshape(-1, 1)
        target_columns = ['formation_energy_ev_natom', 'bandgap_energy_ev']

    # Note: y_true_full is in log1p space (transformed in create_features)
    # OOF predictions are also in log1p space

    model_scores = {}
    for model_name, oof_pred in all_oof_preds.items():
        if isinstance(oof_pred, pd.DataFrame):
            oof_np = oof_pred.values
        elif isinstance(oof_pred, pd.Series):
            oof_np = oof_pred.values.reshape(-1, 1)
        else:
            oof_np = np.array(oof_pred)
            if oof_np.ndim == 1:
                oof_np = oof_np.reshape(-1, 1)

        # Compute RMSE in log space (equivalent to RMSLE in original space)
        # Since both are in log1p space, we compute RMSE directly
        rmse_scores = []
        for col_idx in range(oof_np.shape[1]):
            rmse = np.sqrt(np.mean((oof_np[:, col_idx] - y_true_np[:, col_idx]) ** 2))
            rmse_scores.append(rmse)

        mean_rmse = np.mean(rmse_scores)
        model_scores[model_name] = mean_rmse
        print(f"Model {model_name} - Mean RMSE (log space): {mean_rmse:.6f}")
        for col_idx, col_name in enumerate(target_columns[:oof_np.shape[1]]):
            print(f"  {col_name}: {rmse_scores[col_idx]:.6f}")

    # Step 3: Compute final ensemble
    # For initial iteration, use simple averaging of all models

    if len(aggregated_test_preds) == 0:
        raise ValueError("No model predictions available for ensemble")

    # Get the shape from the first model's predictions
    first_model = list(aggregated_test_preds.keys())[0]
    n_samples = aggregated_test_preds[first_model].shape[0]
    n_targets = aggregated_test_preds[first_model].shape[1] if aggregated_test_preds[first_model].ndim > 1 else 1

    # Simple average ensemble
    ensemble_pred = np.zeros((n_samples, n_targets))

    for model_name, pred in aggregated_test_preds.items():
        if pred.ndim == 1:
            pred = pred.reshape(-1, 1)
        ensemble_pred += pred

    ensemble_pred /= len(aggregated_test_preds)

    # Transform predictions back from log1p space to original scale
    # The targets were transformed using log1p in create_features
    ensemble_pred_original = np.expm1(ensemble_pred)

    # Ensure non-negative predictions (formation energy and bandgap should be >= 0)
    ensemble_pred_original = np.maximum(ensemble_pred_original, 0)

    # Handle any NaN or Infinity values
    ensemble_pred_original = np.nan_to_num(ensemble_pred_original, nan=0.0, posinf=0.0, neginf=0.0)

    # Convert to DataFrame with proper column names
    if n_targets == 2:
        columns = ['formation_energy_ev_natom', 'bandgap_energy_ev']
    else:
        columns = target_columns[:n_targets]

    final_predictions = pd.DataFrame(ensemble_pred_original, columns=columns)

    # Final validation - ensure no NaN or Infinity
    final_predictions = final_predictions.replace([np.inf, -np.inf], np.nan)
    final_predictions = final_predictions.fillna(final_predictions.mean())
    final_predictions = final_predictions.fillna(0)

    # Save OOF predictions and model scores for future iterations
    os.makedirs(OUTPUT_DATA_PATH, exist_ok=True)

    # Save model scores
    scores_path = os.path.join(OUTPUT_DATA_PATH, "model_scores.csv")
    scores_df = pd.DataFrame([
        {"model": model_name, "mean_rmse_log_space": score}
        for model_name, score in model_scores.items()
    ])
    scores_df.to_csv(scores_path, index=False)
    print(f"\nModel scores saved to {scores_path}")

    # Save OOF predictions for potential future ensemble optimization
    for model_name, oof_pred in all_oof_preds.items():
        oof_path = os.path.join(OUTPUT_DATA_PATH, f"oof_preds_{model_name}.csv")
        if isinstance(oof_pred, pd.DataFrame):
            oof_pred.to_csv(oof_path, index=False)
        else:
            pd.DataFrame(oof_pred).to_csv(oof_path, index=False)
    print(f"OOF predictions saved to {OUTPUT_DATA_PATH}")

    print(f"\nFinal ensemble predictions shape: {final_predictions.shape}")
    print(f"Prediction statistics:")
    print(final_predictions.describe())

    return final_predictions
