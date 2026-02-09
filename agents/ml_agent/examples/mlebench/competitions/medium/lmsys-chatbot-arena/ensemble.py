import numpy as np
from sklearn.metrics import log_loss
from typing import Dict, Any

# Concrete type definitions for LMSYS Chatbot Arena
y = np.ndarray           # Ground truth integer labels [0, 1, 2]
Predictions = np.ndarray # Probability matrix of shape (N, 3)

def ensemble(
    all_val_preds: Dict[str, Predictions],
    all_test_preds: Dict[str, Predictions],
    y_val: y,
) -> Predictions:
    """
    Combines predictions from multiple models (folds) into a final superior prediction
    using the arithmetic mean of class probabilities.

    Efficiency: Uses NumPy vectorization for high-performance aggregation.
    Objective: Reduces log-loss by smoothing variance across different model folds.
    """
    if not all_test_preds:
        raise ValueError("Ensemble received empty all_test_preds dictionary.")

    print(f"Starting Ensemble Stage for {len(all_test_preds)} model entries...")

    # 1. Evaluate individual model performance
    # This provides diagnostics on how each fold/model contributed to the ensemble.
    for model_name, val_probs in all_val_preds.items():
        try:
            # Only calculate log-loss if labels and predictions are aligned in size
            if val_probs.shape[0] == y_val.shape[0]:
                score = log_loss(y_val, val_probs)
                print(f"  [Evaluation] Model: {model_name} | Validation Log Loss: {score:.6f}")
            else:
                # This occurs if all_val_preds contains fold-specific OOFs while y_val is full
                # or vice-versa. We skip to avoid crashing the pipeline.
                print(f"  [Evaluation] Model: {model_name} | Size Mismatch (Pred: {len(val_probs)}, Target: {len(y_val)}).")
        except Exception as e:
            print(f"  [Warning] Could not calculate score for {model_name}: {e}")

    # 2. Arithmetic Mean Aggregation
    # We aggregate along axis=0 (the models/folds axis) to get the mean probability for each sample.
    test_preds_list = list(all_test_preds.values())
    
    # Check for consistency in prediction shapes
    expected_shape = test_preds_list[0].shape
    for i, p in enumerate(test_preds_list):
        if p.shape != expected_shape:
            raise ValueError(f"Prediction shape mismatch! Model index {i} has shape {p.shape}, expected {expected_shape}.")

    print(f"Aggregating {len(test_preds_list)} test prediction sets of shape {expected_shape}...")
    
    # Use np.stack and np.mean for efficient calculation using system RAM/CPU
    # Stack shape: (num_models, num_samples, 3)
    # Mean result: (num_samples, 3)
    final_test_preds = np.mean(np.stack(test_preds_list, axis=0), axis=0)

    # 3. Post-processing and Validation
    # Ensure the sum of probabilities per row is 1 (floating point rounding might occur, 
    # but log_loss and competition metrics handle this; we just check for valid numbers)
    if np.isnan(final_test_preds).any() or np.isinf(final_test_preds).any():
        raise ValueError("Ensemble encountered NaN or Infinity values in final predictions.")

    # Log confidence check (optional but helpful for monitoring)
    avg_confidence = np.max(final_test_preds, axis=1).mean()
    print(f"Ensemble completed. Average Max Probability (Confidence): {avg_confidence:.4f}")
    print(f"Final output shape: {final_test_preds.shape}")

    return final_test_preds