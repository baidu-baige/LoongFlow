import numpy as np
from sklearn import metrics
from typing import Dict, List, Any

# Task-adaptive type definitions for multi-label bird classification
y = np.ndarray           # (N, 19) float32 ground truth matrix
Predictions = np.ndarray # (N, 19) float32 probability matrix

def ensemble(
    all_val_preds: Dict[str, Predictions],
    all_test_preds: Dict[str, Predictions],
    y_val: y,
) -> Predictions:
    """
    Aggregates multi-label predictions across folds and models into a final robust output.
    
    This implementation combines the outputs of the Triple-GBDT Ensemble (XGBoost, 
    LightGBM, CatBoost) to maximize generalization and minimize error correlation.
    Following the Technical Specification, this stage performs a weighted arithmetic 
    averaging across 15 model instances (5 folds * 3 architectures). 
    
    Since the upstream 'train_and_predict' stage already applies the 0.4/0.3/0.3 
    weighting for the individual models within each fold, this module executes 
    the final cross-validation fold aggregation.

    Args:
        all_val_preds (Dict[str, Predictions]): Dictionary mapping fold identifiers 
                                               to out-of-fold validation predictions.
        all_test_preds (Dict[str, Predictions]): Dictionary mapping fold identifiers 
                                                to test set predictions.
        y_val (y): Ground truth labels for the training set (multi-label matrix).

    Returns:
        Predictions: Final aggregated test prediction matrix of shape (N_test, 19).
    """
    print("Stage 5: Starting Ensemble Stage (Cross-Fold Aggregation)...")
    
    model_keys = list(all_test_preds.keys())
    if not model_keys:
        raise ValueError("Ensemble Failure: No test predictions provided in all_test_preds.")

    num_folds = len(model_keys)
    print(f"Aggregating predictions from {num_folds} cross-validation folds.")

    # 1. Performance Diagnostics (OOF Multi-Label AUC)
    # The competition metric is Macro-AUC across 19 species.
    # We evaluate individual fold performance if the validation predictions match the label shape.
    for fold_key in model_keys:
        if fold_key in all_val_preds:
            v_preds = all_val_preds[fold_key]
            # Check if validation predictions align with provided ground truth
            if y_val is not None and v_preds.shape == y_val.shape:
                try:
                    fold_auc = metrics.roc_auc_score(y_val, v_preds, average='macro')
                    print(f"Fold '{fold_key}' OOF Macro-AUC: {fold_auc:.4f}")
                except Exception as e:
                    print(f"Diagnostic Warning: Could not calculate AUC for {fold_key}: {e}")

    # 2. Weighted Arithmetic Averaging
    # As per the specification, the final output must reflect a 0.4/0.3/0.3 blend.
    # Because each fold's prediction is already weighted by train_and_predict, 
    # taking the arithmetic mean across folds preserves this global distribution.
    
    test_preds_list = []
    for fold_key in model_keys:
        fold_preds = all_test_preds[fold_key]
        
        # Integrity checks: Ensure no NaNs or Infs propagate to final submission
        if np.isnan(fold_preds).any() or np.isinf(fold_preds).any():
            raise ValueError(f"Data Integrity Error: Model '{fold_key}' contains NaN or Inf values.")
            
        test_preds_list.append(fold_preds.astype(np.float64))

    # Efficient aggregation using NumPy
    # Shape: (N_folds, N_test, 19) -> (N_test, 19)
    stacked_preds = np.stack(test_preds_list, axis=0)
    final_test_preds = np.mean(stacked_preds, axis=0).astype(np.float32)

    # 3. Post-processing and Constraint Enforcement
    # The bird classification task requires probabilities in range [0, 1].
    # Although GBDT outputs are typically bounded, we explicitly clip to guarantee compliance.
    final_test_preds = np.clip(final_test_preds, 0.0, 1.0)
    
    # Final check for numerical consistency
    if np.isnan(final_test_preds).any():
        raise ValueError("Ensemble Failure: NaN values generated during final aggregation.")

    n_test, n_species = final_test_preds.shape
    print(f"Ensemble complete. Generated predictions for {n_test} recordings and {n_species} species.")
    print(f"Final Prediction Stats - Mean: {np.mean(final_test_preds):.4f}, Max: {np.max(final_test_preds):.4f}")
    
    return final_test_preds