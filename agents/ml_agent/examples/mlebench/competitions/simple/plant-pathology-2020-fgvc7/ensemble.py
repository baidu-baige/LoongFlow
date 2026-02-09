import numpy as np
from typing import Dict, List, Any
from sklearn.metrics import roc_auc_score

# Task-adaptive type definitions 
y = np.ndarray           # Target matrix (N, 4)
Predictions = np.ndarray # Probability matrix (N, 4)

def ensemble(
    all_val_preds: Dict[str, Predictions],
    all_test_preds: Dict[str, Predictions],
    y_val: y,
) -> Predictions:
    """
    Combines predictions from multiple models using a weighted average based on OOF ROC AUC scores.
    Aggregates diverse model outputs to maximize column-wise mean ROC AUC.

    Args:
        all_val_preds (Dict[str, Predictions]): Mapping of model names to their OOF predictions.
        all_test_preds (Dict[str, Predictions]): Mapping of model names to their test predictions.
        y_val (y): Ground truth targets for validation (N, 4).

    Returns:
        Predictions: Final aggregated test set predictions (M, 4).
    """
    print("Stage 5: Starting ensemble aggregation...")
    
    if not all_val_preds or not all_test_preds:
        raise ValueError("Ensemble received empty prediction dictionaries.")

    model_names = list(all_val_preds.keys())
    num_models = len(model_names)
    print(f"Ensembling {num_models} models: {model_names}")

    # Step 1: Evaluate individual model scores (Mean Column-wise ROC AUC)
    model_scores = {}
    for name in model_names:
        val_pred = all_val_preds[name]
        
        # Mean column-wise ROC AUC calculation
        # y_val and val_pred are expected to be (N, 4)
        try:
            # Multi-label AUC with macro averaging calculates AUC for each column and then means them.
            score = roc_auc_score(y_val, val_pred, average='macro', multi_class='ovr')
            model_scores[name] = score
            print(f"Model '{name}' OOF ROC AUC: {score:.5f}")
        except Exception as e:
            print(f"Error calculating AUC for model {name}: {e}")
            raise

    # Step 2: Determine Weights
    # If scores are very similar (within 0.001), use equal weighting to reduce variance.
    # Otherwise, use weighted average based on relative performance.
    scores = np.array([model_scores[name] for name in model_names])
    score_range = np.max(scores) - np.min(scores)
    
    if score_range < 0.001:
        print(f"Scores are similar (range {score_range:.5f}). Using equal weighting.")
        weights = np.ones(num_models) / num_models
    else:
        # We use (AUC - 0.5) as the basis for weighting to emphasize performance above chance.
        # We also apply a power of 2 to further reward the best performing models.
        adjusted_scores = np.maximum(scores - 0.5, 0.01)
        power_weights = adjusted_scores ** 2
        weights = power_weights / np.sum(power_weights)
        print("Using weighted averaging based on OOF performance.")

    for i, name in enumerate(model_names):
        print(f"Model: {name:30} | Weight: {weights[i]:.4f} | AUC: {model_scores[name]:.5f}")

    # Step 3: Compute weighted average for test predictions
    # Initialize with the first model's predictions
    first_model = model_names[0]
    final_test_preds = np.zeros_like(all_test_preds[first_model], dtype=np.float32)
    
    for i, name in enumerate(model_names):
        test_pred = all_test_preds[name]
        
        if test_pred.shape != final_test_preds.shape:
            raise ValueError(f"Shape mismatch in test predictions for model {name}. "
                             f"Expected {final_test_preds.shape}, got {test_pred.shape}")
            
        final_test_preds += weights[i] * test_pred

    # Step 4: Final validation and cleanup
    if np.isnan(final_test_preds).any() or np.isinf(final_test_preds).any():
        raise ValueError("Ensemble generated NaN or Infinity values in predictions.")

    # Constraints check: Probabilities should be in [0, 1]
    final_test_preds = np.clip(final_test_preds, 0.0, 1.0)
    
    # Calculate OOF ensemble score for logging
    final_val_preds = np.zeros_like(all_val_preds[first_model], dtype=np.float32)
    for i, name in enumerate(model_names):
        final_val_preds += weights[i] * all_val_preds[name]
    
    ensemble_val_score = roc_auc_score(y_val, final_val_preds, average='macro', multi_class='ovr')
    print(f"Ensemble OOF ROC AUC score: {ensemble_val_score:.5f}")
    
    print(f"Ensemble complete. Final prediction shape: {final_test_preds.shape}")
    
    return final_test_preds