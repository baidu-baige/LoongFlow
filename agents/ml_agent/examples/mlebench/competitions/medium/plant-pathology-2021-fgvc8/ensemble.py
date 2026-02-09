import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from typing import Dict, List, Any
import gc

# Task-adaptive type definitions
y = np.ndarray           # Binary matrix (N, 6)
Predictions = np.ndarray # Final array of strings (N_test,)

def ensemble(
    all_val_preds: Dict[str, np.ndarray],
    all_test_preds: Dict[str, np.ndarray],
    y_val: np.ndarray,
) -> np.ndarray:
    """
    Combines predictions from multiple models using weighted averaging and class-specific 
    threshold optimization to maximize the Mean F1-Score.

    Args:
        all_val_preds (Dict[str, np.ndarray]): Probs from models on validation set.
        all_test_preds (Dict[str, np.ndarray]): Probs from models on test set.
        y_val (np.ndarray): Ground truth binary matrix for validation.

    Returns:
        np.ndarray: Array of space-delimited label strings for each test image.
    """
    print("Initiating Ensemble Engine...")
    
    # Technical Specification: Target Classes (Order must match load_data)
    target_classes = ['scab', 'healthy', 'frog_eye_leaf_spot', 'rust', 'complex', 'powdery_mildew']
    num_classes = len(target_classes)
    
    # 1. Weighted Average Ensemble
    # Specification: 0.5 (ConvNeXt-Large), 0.5 (Swin-Large)
    # Note: If only one model is present in the dictionary, it takes full weight.
    model_names = list(all_val_preds.keys())
    if not model_names:
        raise ValueError("Ensemble Error: No model predictions provided in all_val_preds.")
    
    print(f"Ensembling models: {model_names}")
    
    # Define weights based on specification
    weights = {
        'convnext_large': 0.5,
        'swin_large': 0.5
    }
    
    # Normalize weights for the models actually present
    present_weights = np.array([weights.get(name, 1.0 / len(model_names)) for name in model_names])
    present_weights /= present_weights.sum()
    
    def compute_weighted_average(preds_dict: Dict[str, np.ndarray], weight_arr: np.ndarray) -> np.ndarray:
        avg_preds = None
        for i, name in enumerate(model_names):
            p = preds_dict[name]
            if avg_preds is None:
                avg_preds = p * weight_arr[i]
            else:
                avg_preds += p * weight_arr[i]
        return avg_preds

    val_ensemble_probs = compute_weighted_average(all_val_preds, present_weights)
    test_ensemble_probs = compute_weighted_average(all_test_preds, present_weights)
    
    # 2. Threshold Optimization
    # We optimize thresholds for each class independently to maximize F1-Score on the validation set.
    print("Optimizing class-wise thresholds on validation data...")
    best_thresholds = np.full(num_classes, 0.5, dtype=np.float32)
    
    for i in range(num_classes):
        best_f1 = -1.0
        best_t = 0.5
        # Search space for threshold optimization
        for t in np.linspace(0.05, 0.95, 91):
            current_f1 = f1_score(y_val[:, i], (val_ensemble_probs[:, i] > t).astype(int), zero_division=0)
            if current_f1 > best_f1:
                best_f1 = current_f1
                best_t = t
        best_thresholds[i] = best_t
        print(f"Class '{target_classes[i]}' - Optimal Threshold: {best_t:.3f}, Max F1: {best_f1:.4f}")

    # 3. Apply Optimized Thresholds to Test Set
    print("Applying thresholds and generating final labels...")
    test_binary = (test_ensemble_probs > best_thresholds).astype(int)
    
    # Convert binary matrix to space-delimited string labels
    final_labels = []
    for row in test_binary:
        labels = [target_classes[idx] for idx, val in enumerate(row) if val == 1]
        
        # Post-processing: If no labels are predicted, fall back to the highest probability class
        if not labels:
            highest_prob_idx = np.argmax(test_ensemble_probs[len(final_labels)])
            labels = [target_classes[highest_prob_idx]]
            
        final_labels.append(" ".join(labels))
    
    # Resource Management
    del val_ensemble_probs, test_ensemble_probs, test_binary
    gc.collect()
    
    final_output = np.array(final_labels)
    
    # Validation of output constraints
    num_test_samples = len(next(iter(all_test_preds.values())))
    assert len(final_output) == num_test_samples, f"Alignment Error: Output size {len(final_output)} != Test size {num_test_samples}"
    
    print(f"Ensemble complete. Generated predictions for {len(final_output)} samples.")
    return final_output