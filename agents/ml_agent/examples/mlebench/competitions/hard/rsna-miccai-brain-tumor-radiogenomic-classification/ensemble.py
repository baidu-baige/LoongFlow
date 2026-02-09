import numpy as np
from sklearn.metrics import roc_auc_score
from typing import Dict, List, Any

# Task-adaptive type definitions
# y: 1D NumPy array (N_samples,) representing the MGMT_value ground truth
# Predictions: 1D NumPy array (N_samples,) containing predicted probabilities
y = np.ndarray
Predictions = np.ndarray

def ensemble(
    all_val_preds: Dict[str, Predictions],
    all_test_preds: Dict[str, Predictions],
    y_val: y,
) -> Predictions:
    """
    Combines predictions from multiple models into a final output using a weighted average strategy.
    
    This implementation follows the technical specification to leverage complementary signals 
    from different MRI modalities (FLAIR, T1w, T1wCE, T2w).
    """
    print("Ensemble: Starting model aggregation...")
    
    # 1. Define heuristic weights from technical specification
    # FLAIR is prioritized (0.4) as it often captures edema/tumor extent effectively.
    modality_weights = {
        'FLAIR': 0.4,
        'T1w': 0.2,
        'T1wCE': 0.2,
        'T2w': 0.2
    }
    
    model_names = list(all_test_preds.keys())
    if not model_names:
        raise RuntimeError("Ensemble failed: No model predictions found in all_test_preds.")

    print(f"Ensemble: Models detected for aggregation: {model_names}")

    # 2. Evaluate individual model performance on validation set
    valid_aucs = {}
    for name in model_names:
        try:
            # Ensure predictions are within [0, 1] range for AUC calculation
            v_preds = np.clip(all_val_preds[name], 0, 1)
            auc = roc_auc_score(y_val, v_preds)
            valid_aucs[name] = auc
            print(f"Ensemble: Model '{name}' Validation AUC: {auc:.4f}")
        except Exception as e:
            print(f"Ensemble: Warning - Could not calculate AUC for {name}: {e}")

    # 3. Apply Weighted Average
    # We map the keys in all_test_preds to the modality weights.
    # If a model name doesn't explicitly contain a modality keyword, we use a neutral weight.
    
    aggregated_test_preds = None
    applied_weights_sum = 0.0
    
    for name in model_names:
        test_preds = all_test_preds[name]
        
        # Determine weight for this model
        target_weight = 1.0  # Default fallback
        matched = False
        for mod, weight in modality_weights.items():
            if mod.lower() in name.lower():
                target_weight = weight
                matched = True
                break
        
        # If the model name doesn't match a modality but others do, 
        # we treat it as a generic model with equal contribution to the "rest".
        if not matched and len(model_names) > 1:
            print(f"Ensemble: Model '{name}' did not match specific modality. Using default weight.")
        
        print(f"Ensemble: Applying weight {target_weight:.2f} to model '{name}'")
        
        if aggregated_test_preds is None:
            aggregated_test_preds = test_preds.astype(np.float64) * target_weight
        else:
            aggregated_test_preds += test_preds.astype(np.float64) * target_weight
            
        applied_weights_sum += target_weight

    # 4. Final Normalization & Safety Checks
    if applied_weights_sum > 0:
        final_test_preds = aggregated_test_preds / applied_weights_sum
    else:
        # Emergency fallback to simple mean if weights are zero
        print("Ensemble: Critical - Weights sum to zero. Falling back to simple average.")
        final_test_preds = np.mean(list(all_test_preds.values()), axis=0)

    # Ensure no NaN or Infinity values (could occur if upstream models failed)
    if np.isnan(final_test_preds).any() or np.isinf(final_test_preds).any():
        print("Ensemble: Warning - NaNs/Infs detected in final results. Performing safe imputation.")
        final_test_preds = np.nan_to_num(final_test_preds, nan=0.5, posinf=1.0, neginf=0.0)

    # Ensure probabilities are bounded [0, 1]
    final_test_preds = np.clip(final_test_preds, 0.0, 1.0)

    print(f"Ensemble: Aggregation complete. Output size: {len(final_test_preds)}")
    
    return final_test_preds.astype(np.float32)