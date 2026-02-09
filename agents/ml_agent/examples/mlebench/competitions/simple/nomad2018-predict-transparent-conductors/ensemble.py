import numpy as np
import pandas as pd
from typing import Dict, Any

# Task-adaptive type definitions for NOMAD2018
y = pd.DataFrame      # Ground truth targets for formation and bandgap energy
Predictions = np.ndarray # Model predictions array of shape (n_samples, 2)

def ensemble(
    all_val_preds: Dict[str, Predictions],
    all_test_preds: Dict[str, Predictions],
    y_val: y,
) -> Predictions:
    """
    Consolidates predictions from multiple models into a final output using a 
    weighted average strategy optimized for the RMSLE metric.
    
    Weights are determined based on the mean RMSLE performance across target 
    columns on the validation set. Blending is performed in the log-transformed 
    space to align with the metric's characteristics.
    """
    print(f"Starting Ensemble Module (Stage 5): Processing {len(all_test_preds)} model outputs.")

    # 1. Align models present in both validation and test dictionaries
    model_names = list(set(all_val_preds.keys()) & set(all_test_preds.keys()))
    if not model_names:
        raise KeyError("Ensemble Error: No common model names found between validation and test predictions.")

    # 2. Prepare ground truth for metric calculation
    # Ensure y_val is converted to a numpy array and clipped to non-negative
    y_true = y_val.values if hasattr(y_val, 'values') else np.array(y_val)
    log_true = np.log1p(np.maximum(0, y_true))
    
    # 3. Calculate weights based on individual model performance (Mean RMSLE)
    weights = {}
    for name in model_names:
        val_pred = all_val_preds[name]
        
        # Log-transform predictions for RMSLE calculation
        # Shape is (n_samples, 2) for formation_energy and bandgap_energy
        log_pred = np.log1p(np.maximum(0, val_pred))
        
        # Calculate RMSLE for each column separately
        # Evaluation: sqrt(mean((log(p+1) - log(a+1))^2))
        squared_errors = np.square(log_true - log_pred)
        rmsle_per_col = np.sqrt(np.mean(squared_errors, axis=0))
        mean_rmsle = np.mean(rmsle_per_col)
        
        print(f"  Model Evaluation: '{name}' | Mean RMSLE: {mean_rmsle:.6f}")
        print(f"    - formation_energy_ev_natom RMSLE: {rmsle_per_col[0]:.6f}")
        print(f"    - bandgap_energy_ev RMSLE: {rmsle_per_col[1]:.6f}")
        
        # Use inverse of the error as the weight (higher performance -> higher weight)
        # Small epsilon added for numerical stability
        weights[name] = 1.0 / (mean_rmsle + 1e-12)

    # 4. Normalize weights to sum to 1.0
    total_weight_sum = sum(weights.values())
    normalized_weights = {name: w / total_weight_sum for name, w in weights.items()}
    
    for name, w in normalized_weights.items():
        print(f"  Calculated Weight for '{name}': {w:.4f}")

    # 5. Apply weighted average in log-space
    # This is equivalent to a weighted geometric mean in the (p+1) space,
    # which is robust for skewed distributions and RMSLE objectives.
    first_model_name = model_names[0]
    n_test_samples = all_test_preds[first_model_name].shape[0]
    n_targets = all_test_preds[first_model_name].shape[1]
    
    # Accumulate weighted log-predictions
    final_log_test = np.zeros((n_test_samples, n_targets), dtype=np.float64)
    
    for name in model_names:
        test_preds = all_test_preds[name]
        # Align with log-space optimization
        final_log_test += normalized_weights[name] * np.log1p(np.maximum(0, test_preds))
        
    # 6. Transform back to original feature space
    # exp(log(p+1)) - 1 = p
    final_predictions = np.expm1(final_log_test)
    
    # 7. Quality Assurance
    if np.isnan(final_predictions).any() or np.isinf(final_predictions).any():
        raise ValueError("Critical Error: Ensemble generated non-finite values (NaN/Inf).")
        
    if final_predictions.shape != all_test_preds[first_model_name].shape:
        raise ValueError(f"Shape Mismatch: Ensemble output {final_predictions.shape} "
                         f"does not match input test predictions {all_test_preds[first_model_name].shape}.")

    print(f"Ensemble successfully generated final predictions for {n_test_samples} samples.")
    return final_predictions