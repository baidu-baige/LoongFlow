import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import cohen_kappa_score
from typing import Dict, Any

# Task-adaptive type definitions
# y: numpy array of ground truth integer scores (1-6)
# Predictions: numpy array of continuous or integer model predictions
y = np.ndarray
Predictions = np.ndarray

def ensemble(
    all_val_preds: Dict[str, Predictions],
    all_test_preds: Dict[str, Predictions],
    y_val: y,
) -> Predictions:
    """
    Combines predictions from multiple models using weighted blending and optimizes 
    integer mapping thresholds to maximize the Quadratic Weighted Kappa (QWK) metric.

    Strategy:
    1. Weighted Blending: Prioritize DeBERTa (0.7) and LightGBM (0.3) if identifiable, 
       otherwise use a simple average of all available models.
    2. Threshold Optimization: Find 5 optimal boundaries to map continuous regression 
       outputs to discrete scores [1, 2, 3, 4, 5, 6] using Nelder-Mead optimization.
    """
    print(f"Stage 5: Ensemble starting. Number of models: {len(all_val_preds)}")

    if not all_val_preds:
        raise ValueError("No validation predictions provided for ensembling.")

    # Step 1: Weighted Average Blending
    # Search for DeBERTa and LightGBM keys to apply specified 0.7/0.3 weighting
    deberta_key = next((k for k in all_val_preds.keys() if 'deberta' in k.lower()), None)
    lgbm_key = next((k for k in all_val_preds.keys() if 'lgbm' in k.lower() or 'lightgbm' in k.lower()), None)

    if deberta_key and lgbm_key and deberta_key != lgbm_key:
        print(f"Applying weighted blend: {deberta_key} (0.7) + {lgbm_key} (0.3)")
        combined_val = 0.7 * all_val_preds[deberta_key] + 0.3 * all_val_preds[lgbm_key]
        combined_test = 0.7 * all_test_preds[deberta_key] + 0.3 * all_test_preds[lgbm_key]
    else:
        # Fallback to simple average if specific models aren't found or only one model exists
        print("Model-specific keys not found or unique. Using simple average of available models.")
        combined_val = np.mean(list(all_val_preds.values()), axis=0)
        combined_test = np.mean(list(all_test_preds.values()), axis=0)

    # Step 2: Threshold Optimization for QWK
    # Objective: Maximize QWK by finding optimal boundaries for rounding
    # Initial thresholds for classes [1, 2, 3, 4, 5, 6] are the midpoints
    initial_thresholds = [1.5, 2.5, 3.5, 4.5, 5.5]
    
    def qwk_objective(thresholds: np.ndarray) -> float:
        """Negative QWK for minimization."""
        # Ensure thresholds are sorted to maintain ordinal mapping
        sorted_thresholds = np.sort(thresholds)
        # Digitize maps values to [0, 1, 2, 3, 4, 5], so we add 1 for [1, 2, 3, 4, 5, 6]
        preds = np.digitize(combined_val, sorted_thresholds) + 1
        # Calculate QWK
        score = cohen_kappa_score(y_val, preds, weights='quadratic')
        return -score

    print("Optimizing thresholds using Nelder-Mead...")
    optimization_result = minimize(
        qwk_objective, 
        initial_thresholds, 
        method='Nelder-Mead',
        options={'maxiter': 1000}
    )
    
    optimized_thresholds = np.sort(optimization_result.x)
    final_val_qwk = -optimization_result.fun
    
    print(f"Optimization complete. Best OOF QWK: {final_val_qwk:.4f}")
    print(f"Optimized Thresholds: {optimized_thresholds}")

    # Step 3: Generate Final Test Predictions
    # Apply optimized thresholds to the combined test set predictions
    final_test_preds = np.digitize(combined_test, optimized_thresholds) + 1
    
    # Ensure values are within the valid rubric range [1, 6] and handle any numerical edge cases
    final_test_preds = np.clip(final_test_preds, 1, 6).astype(np.int64)

    # Sanity checks
    if np.isnan(final_test_preds).any():
        raise ValueError("Ensemble generated NaN values in final predictions.")
    
    if len(final_test_preds) != len(combined_test):
        raise ValueError(f"Output length mismatch: Expected {len(combined_test)}, got {len(final_test_preds)}")

    print(f"Final ensemble output generated. Mean prediction: {final_test_preds.mean():.4f}")
    return final_test_preds