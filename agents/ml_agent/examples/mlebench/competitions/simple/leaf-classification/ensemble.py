import numpy as np
import cudf
from typing import Dict, Any
from sklearn.metrics import log_loss
from scipy.optimize import minimize

# Task-adaptive type definitions
y = cudf.Series           # Target vector type: RAPIDS Series (encoded species labels)
Predictions = np.ndarray  # Model predictions type: Probability matrix [N, 99]

BASE_DATA_PATH = "/mnt/pfs/loongflow/devmachine/h20-09/evolux/output/mlebench/leaf-classification/prepared/public"
OUTPUT_DATA_PATH = "output/5e63fe40-52af-4d8b-ac71-4d3a91b9999f/54/executor/output"

def ensemble(
    all_val_preds: Dict[str, Predictions],
    all_test_preds: Dict[str, Predictions],
    y_val: y,
) -> Predictions:
    """
    Combines model predictions using regularized SLSQP weight optimization and 
    global temperature scaling to minimize multi-class log-loss.

    Technical Specification:
    - Weight Optimization: Minimize log_loss(y_true, weighted_probs) + 0.01 * sum(w**2).
    - Temperature Scaling: Grid search T in [0.5, 3.5] with step 0.05.
    - Normalization: Clip final probabilities to 1e-7 and row-normalize.
    """
    print("Ensemble Stage: Initializing global calibration and fusion pipeline.")

    # Convert GPU-backed targets to CPU for optimization context
    if hasattr(y_val, 'to_numpy'):
        y_true = y_val.to_numpy().astype(np.int32)
    else:
        y_true = np.array(y_val).astype(np.int32)

    model_names = sorted(list(all_val_preds.keys()))
    if not model_names:
        raise ValueError("Ensemble Stage Error: No model predictions found.")

    # Consolidation and precision alignment (using float64 for optimization stability)
    val_list = [all_val_preds[name].astype(np.float64) for name in model_names]
    test_list = [all_test_preds[name].astype(np.float64) for name in model_names]
    
    n_models = len(model_names)
    n_classes = 99
    eps = 1e-7
    all_labels = np.arange(n_classes)

    def finalize_row_probs(probs: np.ndarray, clip_val: float) -> np.ndarray:
        """Ensures probability matrix integrity via clipping and row-normalization."""
        p = np.clip(probs, clip_val, 1.0 - clip_val)
        row_sums = p.sum(axis=1, keepdims=True)
        # Handle potential zero sums
        row_sums[row_sums == 0] = 1.0
        return p / row_sums

    # --- Step 1: Regularized Weight Optimization (SLSQP) ---
    print(f"Ensemble Stage: Optimizing weights for {n_models} models with L2 regularization (0.01)...")
    
    def weight_objective(weights):
        w = np.array(weights)
        # Compute weighted average of probabilities
        ensemble_val = np.zeros_like(val_list[0])
        for i in range(n_models):
            ensemble_val += w[i] * val_list[i]
            
        # Integrity check for log-loss calculation
        ensemble_val = finalize_row_probs(ensemble_val, eps)
        
        # Loss formula: Multi-class log-loss + L2 Weight Penalty
        loss = log_loss(y_true, ensemble_val, labels=all_labels)
        reg = 0.01 * np.sum(w**2)
        return loss + reg

    # Constraints: Weights must sum to 1.0
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
    # Bounds: Weights must be in range [0, 1]
    bounds = [(0, 1)] * n_models
    initial_weights = np.ones(n_models) / n_models
    
    res = minimize(
        weight_objective, 
        initial_weights, 
        method='SLSQP', 
        bounds=bounds, 
        constraints=constraints,
        options={'maxiter': 1000, 'ftol': 1e-12}
    )
    
    if not res.success:
        print(f"Ensemble Stage Warning: Optimization did not converge fully: {res.message}")

    best_weights = np.clip(res.x, 0, 1)
    best_weights /= (best_weights.sum() + 1e-18)
    
    for i, name in enumerate(model_names):
        print(f" - Model '{name}' optimized weight: {best_weights[i]:.6f}")

    # Aggregate OOF and Test predictions using optimized weights
    p_val_weighted = np.zeros_like(val_list[0])
    p_test_weighted = np.zeros_like(test_list[0])
    for i in range(n_models):
        p_val_weighted += best_weights[i] * val_list[i]
        p_test_weighted += best_weights[i] * test_list[i]

    # --- Step 2: Global Temperature Scaling Optimization ---
    print("Ensemble Stage: Grid-searching optimal Temperature T in [0.5, 3.5]...")
    
    best_T = 1.0
    min_loss = float('inf')
    T_candidates = np.arange(0.5, 3.51, 0.05)
    
    for T in T_candidates:
        # Temperature Transformation: p' = normalize(p^(1/T))
        # T < 1 sharpens the distribution; T > 1 softens it.
        p_temp = np.power(p_val_weighted, 1.0 / T)
        p_temp = finalize_row_probs(p_temp, eps)
        
        current_loss = log_loss(y_true, p_temp, labels=all_labels)
        if current_loss < min_loss:
            min_loss = current_loss
            best_T = T

    print(f"Ensemble Stage: Optimal Temperature T = {best_T:.2f} (OOF Cross-Entropy: {min_loss:.6f})")

    # --- Step 3: Final Calibration and Integrity Check ---
    # Apply the optimized temperature scaling to the weighted test predictions
    final_test_preds = np.power(p_test_weighted, 1.0 / best_T)
    final_test_preds = finalize_row_probs(final_test_preds, eps)

    # Validate output integrity
    if np.isnan(final_test_preds).any() or np.isinf(final_test_preds).any():
        raise ValueError("Ensemble Stage Failure: Non-finite values detected in final calibrated predictions.")
    
    expected_rows = test_list[0].shape[0]
    if final_test_preds.shape != (expected_rows, n_classes):
        raise ValueError(f"Ensemble Stage Failure: Shape mismatch. Expected {(expected_rows, n_classes)}, got {final_test_preds.shape}")

    print(f"Ensemble Stage Complete. Generated calibration-fused matrix of shape {final_test_preds.shape}")
    
    return final_test_preds.astype(np.float32)