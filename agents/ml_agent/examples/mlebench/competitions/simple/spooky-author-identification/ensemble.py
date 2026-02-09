import numpy as np
import pandas as pd
from typing import Dict, List, Any
from sklearn.metrics import log_loss
from scipy.optimize import minimize

# Task-adaptive type definitions
y = np.ndarray           # Integer-encoded author labels [0, 1, 2]
Predictions = np.ndarray # Probability matrix of shape (N, 3)

BASE_DATA_PATH = "/mnt/pfs/loongflow/devmachine/1-04/evolux/output/mlebench/spooky-author-identification/prepared/public"
OUTPUT_DATA_PATH = "output/368bc6e8-482c-48b8-a870-040b0c3a264c/6/executor/output"

def ensemble(
    all_val_preds: Dict[str, Predictions],
    all_test_preds: Dict[str, Predictions],
    y_val: y,
) -> Predictions:
    """
    Optimizes the blending of a 5-pillar ensemble using SLSQP to minimize multi-class log-loss.
    The ensemble combines DeBERTa-v3-large, RoBERTa-large, Logistic Regression, 
    Multinomial Naive Bayes, and LightGBM.

    Args:
        all_val_preds (Dict[str, Predictions]): Model name to OOF probability matrix.
        all_test_preds (Dict[str, Predictions]): Model name to test probability matrix.
        y_val (y): Ground truth integer labels (0: EAP, 1: HPL, 2: MWS).

    Returns:
        Predictions: Ensembled test set probability matrix.
    """
    print("Starting ensemble stage: Weighted Average Optimization (SLSQP).")

    # 1. Component Verification
    # Expected models from the Technical Specification
    expected_models = [
        "deberta_v3_large", 
        "roberta_large", 
        "logistic_regression", 
        "multinomial_nb", 
        "lightgbm_meta"
    ]
    
    # Check if all expected models are present
    available_models = list(all_val_preds.keys())
    for model in expected_models:
        if model not in available_models:
            print(f"Warning: Expected model '{model}' not found. Ensemble will proceed with: {available_models}")

    # Use the intersection of expected and actually provided models to maintain order
    active_models = [m for m in expected_models if m in available_models]
    if not active_models:
        # Fallback to whatever is available if none of the expected are found
        active_models = available_models
        if not active_models:
            raise ValueError("Ensemble error: No model predictions provided.")

    val_preds_list = [all_val_preds[m] for m in active_models]
    test_preds_list = [all_test_preds[m] for m in active_models]

    # Integrity Check: Ensure no NaNs or Infs
    for i, name in enumerate(active_models):
        if np.isnan(val_preds_list[i]).any() or np.isnan(test_preds_list[i]).any():
            raise ValueError(f"Ensemble error: NaN values detected in predictions for model '{name}'.")
        
        # Calculate individual model log-loss for logging
        loss = log_loss(y_val, np.clip(val_preds_list[i], 1e-15, 1 - 1e-15))
        print(f"  - Model: {name:<20} | OOF Log-Loss: {loss:.6f}")

    # 2. Weight Optimization via SLSQP
    def objective_func(weights):
        # Calculate weighted average of probabilities
        ensemble_val = np.zeros_like(val_preds_list[0])
        for i in range(len(val_preds_list)):
            ensemble_val += weights[i] * val_preds_list[i]
            
        # Clipping per competition rules to avoid log-loss infinity
        ensemble_val = np.clip(ensemble_val, 1e-15, 1 - 1e-15)
        
        # Row-normalization to ensure valid probability distribution
        ensemble_val = ensemble_val / ensemble_val.sum(axis=1, keepdims=True)
        
        return log_loss(y_val, ensemble_val)

    # Initial Guess: Uniform distribution
    num_models = len(active_models)
    initial_weights = np.ones(num_models) / num_models
    
    # Constraints: Weights sum to 1.0
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
    
    # Bounds: Weights between 0.0 and 1.0
    bounds = [(0.0, 1.0) for _ in range(num_models)]

    print(f"Optimizing weights for {num_models} models...")
    res = minimize(
        objective_func,
        initial_weights,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'ftol': 1e-12, 'maxiter': 1000}
    )

    if not res.success:
        # Optimization failure is critical
        raise RuntimeError(f"Weight optimization failed: {res.message}")

    optimal_weights = res.x
    print("Optimal Weights Found:")
    for name, weight in zip(active_models, optimal_weights):
        print(f"  - {name:<20}: {weight:.6f}")

    # 3. Generate Final Ensembled Predictions
    print("Generating final weighted test predictions...")
    final_test_preds = np.zeros_like(test_preds_list[0])
    for i in range(len(test_preds_list)):
        final_test_preds += optimal_weights[i] * test_preds_list[i]

    # 4. Post-processing
    # Clipping to [1e-15, 1 - 1e-15] as per multi-class log-loss requirements
    final_test_preds = np.clip(final_test_preds, 1e-15, 1 - 1e-15)
    
    # Final row normalization to ensure probabilities sum to 1.0
    final_test_preds = final_test_preds / final_test_preds.sum(axis=1, keepdims=True)

    # Integrity Validation
    if np.isnan(final_test_preds).any():
        raise ValueError("Critical error: Ensemble output contains NaNs.")
    
    final_oof_score = objective_func(optimal_weights)
    print(f"Ensemble Complete. Final Optimized OOF Log-Loss: {final_oof_score:.6f}")

    return final_test_preds