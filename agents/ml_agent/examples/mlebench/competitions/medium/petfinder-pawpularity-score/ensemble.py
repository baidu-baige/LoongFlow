import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from typing import Dict, List, Any

# Task-adaptive type definitions
y = pd.DataFrame           # Target vector type (containing Pawpularity_scaled)
Predictions = np.ndarray    # Model predictions type (numpy arrays)

def ensemble(
    all_val_preds: Dict[str, Predictions],
    all_test_preds: Dict[str, Predictions],
    y_val: y,
) -> Predictions:
    """
    Combines predictions from multiple models into a final output using weighted averaging.
    
    Weights: 
        - ConvNeXt-Large: 0.6
        - Swin-Large: 0.4
    
    This implementation extracts individual model predictions from the upstream 'hybrid_convnext_swin'
    output and calculates the weighted ensemble for both validation (for logging) and test sets.
    """
    print("Starting ensemble stage...")
    
    # The upstream stage 'train_and_predict' provides a single key 'hybrid_convnext_swin'
    # containing a 2-column numpy array: [ConvNeXt_preds, Swin_preds]
    model_key = 'hybrid_convnext_swin'
    if model_key not in all_val_preds or model_key not in all_test_preds:
        raise KeyError(f"Expected predictions for '{model_key}' not found in input dictionaries.")

    val_preds_mt = all_val_preds[model_key]
    test_preds_mt = all_test_preds[model_key]

    # Verification of shapes
    if val_preds_mt.shape[1] != 2 or test_preds_mt.shape[1] != 2:
        raise ValueError(f"Expected 2 columns (ConvNeXt, Swin) in predictions, found {val_preds_mt.shape[1]}.")

    # Extract individual model predictions
    # Based on train_and_predict.py model_configs, index 0 is ConvNeXt, index 1 is Swin
    val_convnext = val_preds_mt[:, 0]
    val_swin = val_preds_mt[:, 1]
    
    test_convnext = test_preds_mt[:, 0]
    test_swin = test_preds_mt[:, 1]

    # Ground truth for validation (scaled back to [0, 100])
    y_true = y_val['Pawpularity_scaled'].values * 100.0

    # Step 1: Evaluate individual model scores
    rmse_convnext = np.sqrt(mean_squared_error(y_true, val_convnext))
    rmse_swin = np.sqrt(mean_squared_error(y_true, val_swin))
    correlation = np.corrcoef(val_convnext, val_swin)[0, 1]

    print(f"Individual Model Performance (Validation RMSE):")
    print(f" - ConvNeXt-Large: {rmse_convnext:.4f}")
    print(f" - Swin-Large:     {rmse_swin:.4f}")
    print(f" - Prediction Correlation: {correlation:.4f}")

    # Step 2: Apply ensemble strategy (Weighted Average)
    # Weights from Technical Specification: 0.6 * ConvNeXt + 0.4 * Swin
    w_convnext = 0.6
    w_swin = 0.4
    
    val_ensemble = (w_convnext * val_convnext) + (w_swin * val_swin)
    test_ensemble = (w_convnext * test_convnext) + (w_swin * test_swin)

    # Step 3: Evaluate ensemble performance
    rmse_ensemble = np.sqrt(mean_squared_error(y_true, val_ensemble))
    print(f"Final Ensemble Validation RMSE: {rmse_ensemble:.4f}")
    
    # Log RMSE improvement
    best_single_rmse = min(rmse_convnext, rmse_swin)
    improvement = best_single_rmse - rmse_ensemble
    print(f"Ensemble Improvement over best single model: {improvement:.4f}")

    # Step 4: Post-processing and Sanity Checks
    # Clip predictions to valid Pawpularity range [1, 100] as per competition rules
    test_ensemble = np.clip(test_ensemble, 1.0, 100.0)

    # Check for NaN or Infinity
    if np.any(np.isnan(test_ensemble)) or np.any(np.isinf(test_ensemble)):
        raise ValueError("Ensemble output contains NaN or Infinity values.")

    print(f"Ensemble complete. Final test predictions shape: {test_ensemble.shape}")
    
    return test_ensemble