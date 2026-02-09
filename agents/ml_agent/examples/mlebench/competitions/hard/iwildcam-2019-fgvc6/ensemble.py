from typing import Dict, Any
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

# Concrete type definitions for the iWildCam 2019 task
y = pd.Series           # Ground truth targets
Predictions = np.ndarray # Probability matrices (N, 23) or Final Class Indices (N,)

def ensemble(
    all_val_preds: Dict[str, Predictions],
    all_test_preds: Dict[str, Predictions],
    y_val: y,
) -> Predictions:
    """
    Combines predictions from multiple models (or folds/TTA versions) into a final output.
    Uses probability averaging (soft voting) to improve robustness and generalization.

    Args:
        all_val_preds: Dictionary mapping model/fold names to their validation probability matrices.
        all_test_preds: Dictionary mapping model/fold names to their test probability matrices.
        y_val: Ground truth targets for validation performance tracking.

    Returns:
        Predictions: A 1D numpy array of final class indices (0-22) for the test set.
    """
    print(f"Starting ensemble process with {len(all_test_preds)} prediction sets...")

    # Step 1: Evaluate individual model scores
    # This provides diagnostics on how each fold/model performed on its respective validation slice.
    # Note: We assume y_val is indexed consistently with the val_preds if they are OOF.
    for model_name, val_probs in all_val_preds.items():
        try:
            # Check if dimensions match for evaluation
            if len(val_probs) == len(y_val):
                val_classes = np.argmax(val_probs, axis=1)
                score = f1_score(y_val, val_classes, average='macro')
                print(f"Validation Score - Model [{model_name}]: Macro F1 = {score:.4f}")
            else:
                # In cross-validation, val_probs might only be for a single fold. 
                # We skip F1 calculation if indices don't match the full y_val.
                pass
        except Exception as e:
            print(f"Warning: Could not evaluate validation score for {model_name}: {e}")

    # Step 2: Apply ensemble strategy - Probability Averaging
    # We aggregate the softmax outputs from all provided models (folds and/or TTA views).
    # Simple averaging is robust against outliers in individual model predictions.
    
    test_probs_list = list(all_test_preds.values())
    if not test_probs_list:
        raise ValueError("The 'all_test_preds' dictionary is empty. No predictions to ensemble.")

    # Stack all probability matrices: (Num_Models, Num_Samples, Num_Classes)
    all_probs_stacked = np.stack(test_probs_list, axis=0)
    
    # Calculate the mean probability across models/folds/TTA
    # Shape: (Num_Samples, Num_Classes)
    mean_probs = np.mean(all_probs_stacked, axis=0)
    
    # Step 3: Generate final class indices
    # We select the category with the highest average probability.
    final_test_preds = np.argmax(mean_probs, axis=1).astype(np.int64)

    # Step 4: Verification
    if np.isnan(final_test_preds).any():
        raise ValueError("Ensemble generated NaN values in final predictions.")
    
    expected_samples = len(test_probs_list[0])
    if len(final_test_preds) != expected_samples:
        raise ValueError(f"Ensemble output size {len(final_test_preds)} does not match input size {expected_samples}.")

    print(f"Ensemble complete. Final prediction shape: {final_test_preds.shape}")
    
    return final_test_preds