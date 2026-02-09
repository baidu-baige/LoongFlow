import numpy as np
import pandas as pd
from typing import Dict, List, Any, Union
from scipy.optimize import minimize
from sklearn.metrics import cohen_kappa_score

# Task-adaptive type definitions
y = np.ndarray  # Diagnosis labels (0-4)
Predictions = np.ndarray  # Regression scores or discrete labels

BASE_DATA_PATH = "/root/workspace/evolux_ml/output/mlebench/aptos2019-blindness-detection/prepared/public"
OUTPUT_DATA_PATH = "output/16326c74-72ad-4b59-ad28-cc76a3d9d373/5/executor/output"

def ensemble(
    all_val_outputs: Dict[str, Predictions],
    all_test_outputs: Dict[str, Predictions],
    y_val: y
) -> Predictions:
    """
    Combines regression predictions from multiple models/folds using arithmetic averaging 
    and optimizes thresholds globally to maximize the Quadratic Weighted Kappa.

    Args:
        all_val_outputs (Dict[str, Predictions]): OOF regression scores from each fold/model.
        all_test_outputs (Dict[str, Predictions]): Test set regression scores from each fold/model.
        y_val (y): Ground truth labels for threshold optimization.

    Returns:
        Predictions: Final discrete diagnosis predictions for the test set.
    """

    # Step 1: Aggregate predictions across folds
    # Convert dictionary values to a list of arrays for aggregation
    val_scores_list = [np.asarray(v).flatten() for v in all_val_outputs.values()]
    test_scores_list = [np.asarray(v).flatten() for v in all_test_outputs.values()]

    if not val_scores_list or not test_scores_list:
        raise ValueError("Ensemble received empty prediction dictionaries.")

    # Calculate arithmetic mean across folds to produce a single stable regression score per image
    # For OOF, this assumes either aligned full-length arrays or that the workflow provides 
    # multiple predictions per instance to be averaged.
    avg_val_scores = np.mean(val_scores_list, axis=0)
    avg_test_scores = np.mean(test_scores_list, axis=0)

    # Refinement: Ensure both OOF and Test scores are clipped to [0, 4] before optimization
    # This maintains consistency with the 0-4 diagnosis scale.
    avg_val_scores = np.clip(avg_val_scores, 0.0, 4.0)
    avg_test_scores = np.clip(avg_test_scores, 0.0, 4.0)

    # Step 2: Global Threshold Optimization
    # Use the pooled OOF regression scores and ground truth labels to find optimal thresholds.
    y_true = np.asarray(y_val).astype(np.int64)
    
    # Strict alignment check: OOF scores must match the length of the ground truth provided
    if len(avg_val_scores) != len(y_true):
        # In some pipeline configurations, targets_full might be the entire dataset while 
        # val_outputs only covers a subset (e.g., validation_mode). 
        # We align to the smaller length to allow optimization to proceed on available data.
        y_true = y_true[:len(avg_val_scores)]

    def kappa_loss(thresholds: List[float], y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Negative Quadratic Weighted Kappa for minimization."""
        # Ensure thresholds are sorted for np.digitize consistency
        t = sorted(thresholds)
        # np.digitize maps continuous scores to [0, 1, 2, 3, 4] based on 4 boundaries
        y_digitized = np.digitize(y_pred, t)
        # We clip to [0, 4] as a safety measure for the metric calculation
        y_digitized = np.clip(y_digitized, 0, 4)
        return -cohen_kappa_score(y_true, y_digitized, weights='quadratic')

    # Initial boundaries: standard rounding points for 0, 1, 2, 3, 4
    init_thresholds = [0.5, 1.5, 2.5, 3.5]
    
    # Optimize thresholds using the Nelder-Mead simplex algorithm
    # This is efficient for low-dimensional non-convex optimization like discrete QWK.
    optimization_result = minimize(
        kappa_loss,
        init_thresholds,
        args=(y_true, avg_val_scores),
        method='Nelder-Mead',
        options={'xatol': 1e-4, 'fatol': 1e-4}
    )

    if not optimization_result.success:
        print(f"Warning: Threshold optimization did not converge: {optimization_result.message}")

    best_thresholds = sorted(optimization_result.x)
    print(f"Optimized Ensemble Thresholds: {best_thresholds}")

    # Step 3: Final Mapping and Discretization
    # Apply the optimized thresholds to the averaged test scores
    final_test_preds = np.digitize(avg_test_scores, best_thresholds)
    
    # Final clipping to guarantee discrete [0, 4] range compliance
    final_test_preds = np.clip(final_test_preds, 0, 4).astype(np.int64)

    # Step 4: Quality Validation
    if np.isnan(final_test_preds).any() or np.isinf(final_test_preds).any():
        raise ValueError("Ensemble generated invalid (NaN/Inf) predictions.")
    
    if len(final_test_preds) != len(avg_test_scores):
        raise ValueError(f"Output size mismatch: {len(final_test_preds)} vs {len(avg_test_scores)}")

    return final_test_preds