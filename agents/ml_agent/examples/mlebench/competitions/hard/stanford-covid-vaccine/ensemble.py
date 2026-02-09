from typing import Dict, List, Any
import numpy as np
import pandas as pd
import os

# Task-adaptive type definitions
# y is a NumPy array of shape (N, 107, 5) containing target values
# Predictions is a NumPy array of shape (N, 107, 5) containing predicted values
y = np.ndarray
Predictions = np.ndarray

BASE_DATA_PATH = "/mnt/pfs/loongflow/devmachine/h20-11/evolux/output/mlebench/stanford-covid-vaccine/prepared/public"
OUTPUT_DATA_PATH = "output/cd1762a7-cbef-43b4-bbfc-bc919a0a7546/1/executor/output"

def mcrmse_metric(y_true: np.ndarray, y_pred: np.ndarray, scored_len: int = 68) -> float:
    """
    Calculates the Mean Columnwise Root Mean Squared Error.
    
    Args:
        y_true: Ground truth array (N, 107, 5)
        y_pred: Prediction array (N, 107, 5)
        scored_len: Number of bases scored (usually 68)
    """
    # Slice to only scored positions
    y_true_scored = y_true[:, :scored_len, :]
    y_pred_scored = y_pred[:, :scored_len, :]
    
    # Calculate RMSE for each target column for each sample
    # Resulting shape: (N, 5)
    mse = np.mean((y_true_scored - y_pred_scored) ** 2, axis=1)
    rmse = np.sqrt(mse)
    
    # Mean across columns and then mean across samples
    return np.mean(rmse)

def ensemble(
    all_val_preds: Dict[str, Predictions],
    all_test_preds: Dict[str, Predictions],
    y_val: y,
) -> Predictions:
    """
    Combines predictions from multiple models into a final output using simple averaging.

    Args:
        all_val_preds (Dict[str, Predictions]): Dictionary mapping model names to their 
                                               out-of-fold predictions (N_val, 107, 5).
        all_test_preds (Dict[str, Predictions]): Dictionary mapping model names to their 
                                                aggregated test predictions (N_test, 107, 5).
        y_val (y): Ground truth targets for validation (N_val, 107, 5).

    Returns:
        Predictions: Final test set predictions (N_test, 107, 5).
    """
    print("Starting ensemble process...")

    if not all_test_preds:
        raise ValueError("No test predictions provided for ensemble.")

    model_names = list(all_test_preds.keys())
    print(f"Ensembling {len(model_names)} models: {model_names}")

    # 1. Evaluate individual model performance on validation set
    if all_val_preds and y_val is not None:
        print("Evaluating individual model performance (Validation MCRMSE):")
        for name in model_names:
            if name in all_val_preds:
                val_score = mcrmse_metric(y_val, all_val_preds[name])
                print(f" - {name}: {val_score:.5f}")
            else:
                print(f" - {name}: Validation predictions missing.")

    # 2. Compute Ensemble of Validation Predictions (for monitoring)
    if all_val_preds and y_val is not None:
        val_pred_list = [all_val_preds[name] for name in model_names if name in all_val_preds]
        if val_pred_list:
            ensemble_val_preds = np.mean(val_pred_list, axis=0)
            ensemble_val_score = mcrmse_metric(y_val, ensemble_val_preds)
            print(f"Ensemble Validation MCRMSE: {ensemble_val_score:.5f}")

    # 3. Simple Averaging of Test Predictions
    # Convert dictionary values to a list and stack them for averaging
    test_preds_list = [all_test_preds[name] for name in model_names]
    
    # Calculate Mean across the model dimension (axis 0)
    # Shape: (N_models, N_test, 107, 5) -> (N_test, 107, 5)
    final_test_preds = np.mean(test_preds_list, axis=0)

    # 4. Integrity Checks
    if np.isnan(final_test_preds).any():
        raise ValueError("Ensemble result contains NaN values.")
    
    if np.isinf(final_test_preds).any():
        raise ValueError("Ensemble result contains Infinity values.")

    # 5. Correlation Analysis (Optional Diagnostic)
    if len(model_names) > 1:
        print("Test prediction correlation between models (averaged over samples/positions/targets):")
        for i in range(len(model_names)):
            for j in range(i + 1, len(model_names)):
                m1 = model_names[i]
                m2 = model_names[j]
                corr = np.corrcoef(all_test_preds[m1].flatten(), all_test_preds[m2].flatten())[0, 1]
                print(f" - Correlation {m1} vs {m2}: {corr:.5f}")

    print(f"Ensemble complete. Final prediction shape: {final_test_preds.shape}")
    
    return final_test_preds