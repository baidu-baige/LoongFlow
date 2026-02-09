import numpy as np
import torch
from typing import Dict, Any, List
from scipy.optimize import minimize
from sklearn.metrics import accuracy_score

# Task-adaptive type definitions
y = torch.Tensor  # Target vector type (labels)
Predictions = np.ndarray  # Probability matrices (N, 5) or Label vectors (N,)

def _flatten_predictions(preds_dict: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """
    Recursively flattens a potentially nested dictionary of predictions into a 
    single-level dictionary mapping unique model/sub-model names to probability matrices.
    """
    flat_preds = {}
    for key, value in preds_dict.items():
        if isinstance(value, dict):
            nested = _flatten_predictions(value)
            for n_key, n_value in nested.items():
                flat_preds[f"{key}_{n_key}"] = n_value
        elif isinstance(value, np.ndarray):
            flat_preds[key] = value
        else:
            # If it's some other type (like a list from DDP), try converting to numpy
            try:
                flat_preds[key] = np.array(value)
            except:
                continue
    return flat_preds

def ensemble(
    all_val_preds: Dict[str, Any],
    all_test_preds: Dict[str, Any],
    y_val: y,
) -> Predictions:
    """
    Combines predictions from multiple models using Weighted Average Ensembling.
    Optimizes weights per-model and per-class using Nelder-Mead on OOF data.

    Args:
        all_val_preds: Nested or flat dict of OOF probability matrices.
        all_test_preds: Nested or flat dict of test probability matrices.
        y_val: Ground truth labels (torch.Tensor).

    Returns:
        Predictions: Final test set labels (1D numpy array).
    """
    print("Starting precision architectural weighting ensemble...")

    # Step 1: Flatten nested dictionaries from upstream stages
    # Upstream 'train_and_predict' returns a dict of models, which might be wrapped 
    # in another dict by the pipeline executor.
    val_flat = _flatten_predictions(all_val_preds)
    test_flat = _flatten_predictions(all_test_preds)

    # Ensure keys match between val and test
    common_keys = sorted(list(set(val_flat.keys()) & set(test_flat.keys())))
    if not common_keys:
        raise ValueError(f"No matching keys found between validation ({val_flat.keys()}) and test ({test_flat.keys()}) predictions.")

    print(f"Ensembling models: {common_keys}")

    # Prepare ground truth
    if isinstance(y_val, torch.Tensor):
        y_true = y_val.detach().cpu().numpy()
    else:
        y_true = np.array(y_val)

    # Collect prediction matrices
    val_probs_list = [val_flat[k] for k in common_keys]
    test_probs_list = [test_flat[k] for k in common_keys]
    
    num_models = len(common_keys)
    num_classes = val_probs_list[0].shape[1]
    num_test_samples = test_probs_list[0].shape[0]

    # Step 2: Define Optimization Objective
    # Optimize a weight matrix of shape (num_models, num_classes)
    def objective(flat_weights: np.ndarray) -> float:
        weights = flat_weights.reshape(num_models, num_classes)
        
        # Calculate weighted probabilities for each class
        # Prob(class c) = sum over models m [ weight(m, c) * Prob(m, class c) ]
        ensemble_val_probs = np.zeros_like(val_probs_list[0])
        for m in range(num_models):
            ensemble_val_probs += val_probs_list[m] * weights[m]
        
        # argmax for labels
        preds = np.argmax(ensemble_val_probs, axis=1)
        
        # Scipy minimize: maximize accuracy -> minimize negative accuracy
        return -accuracy_score(y_true, preds)

    # Step 3: Run Nelder-Mead Optimization
    initial_weights = np.ones(num_models * num_classes)
    
    print(f"Optimizing {len(initial_weights)} weights via Nelder-Mead...")
    
    opt_result = minimize(
        objective,
        initial_weights,
        method='Nelder-Mead',
        options={'maxiter': 1000, 'xatol': 1e-4, 'fatol': 1e-4, 'disp': False}
    )

    best_weights = opt_result.x.reshape(num_models, num_classes)
    print(f"Optimization complete. Best OOF Accuracy: {-opt_result.fun:.5f}")

    # Step 4: Apply optimized weights to test predictions
    final_test_probs = np.zeros((num_test_samples, num_classes), dtype=np.float32)
    for m in range(num_models):
        final_test_probs += test_probs_list[m] * best_weights[m]

    # Step 5: Final Label Generation
    final_test_labels = np.argmax(final_test_probs, axis=1).astype(np.int64)

    # Post-processing validation
    if np.isnan(final_test_labels).any():
        raise ValueError("NaN detected in final ensemble labels.")
    
    print(f"Ensemble generated labels for {len(final_test_labels)} test samples.")
    return final_test_labels