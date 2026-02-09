import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from typing import Dict, List, Any

# Task-adaptive type definitions
y = torch.Tensor  # Target vector type as defined in upstream preprocess
Predictions = np.ndarray  # Model predictions type as defined in upstream train_and_predict


def ensemble(
    all_val_preds: Dict[str, Predictions],
    all_test_preds: Dict[str, Predictions],
    y_val: y,
) -> Predictions:
    """
    Combines predictions from multiple models (folds) into a final output using 
    simple arithmetic averaging as per the technical specification.

    Args:
        all_val_preds (Dict[str, Predictions]): Dictionary mapping model/fold names to 
                                               their out-of-fold predictions.
        all_test_preds (Dict[str, Predictions]): Dictionary mapping model/fold names to 
                                                their test set predictions.
        y_val (y): Ground truth targets for validation (torch.Tensor).

    Returns:
        Predictions: Final test set probabilities (np.ndarray).
    """
    print(f"Ensembling predictions from {len(all_test_preds)} model outputs...")

    # Step 1: Evaluate individual model scores (Log AUC for each fold/model)
    # Convert y_val to numpy for metric calculation
    y_val_np = y_val.cpu().numpy()

    for name, val_p in all_val_preds.items():
        try:
            # Note: This assumes val_p and y_val_np correspond to the same samples.
            # In a K-Fold context, y_val might only correspond to the last fold's labels 
            # or the full OOF labels depending on the workflow implementation.
            if val_p.shape == y_val_np.shape:
                score = roc_auc_score(y_val_np, val_p)
                print(f"Model {name} Validation ROC AUC: {score:.5f}")
            else:
                print(f"Skipping AUC calculation for {name}: Shape mismatch ({val_p.shape} vs {y_val_np.shape})")
        except Exception as e:
            print(f"Could not calculate ROC AUC for {name}: {e}")

    # Step 2: Apply ensemble strategy (Arithmetic Mean)
    # Ensure there are models to ensemble
    if not all_test_preds:
        raise ValueError("No test predictions provided for ensembling.")

    # Collect all test prediction arrays
    test_preds_list = list(all_test_preds.values())

    # Check consistency of prediction shapes
    expected_shape = test_preds_list[0].shape
    for i, p in enumerate(test_preds_list):
        if p.shape != expected_shape:
            raise ValueError(f"Shape mismatch in test predictions for model {list(all_test_preds.keys())[i]}. "
                             f"Expected {expected_shape}, got {p.shape}.")

    # Compute arithmetic mean across all models (folds)
    # Using np.stack and np.mean is memory efficient and leverages vectorization
    final_test_preds = np.mean(np.stack(test_preds_list), axis=0)

    # Step 3: Integrity check
    if np.isnan(final_test_preds).any() or np.isinf(final_test_preds).any():
        raise ValueError("Ensemble produced NaN or Infinity values.")

    print(f"Ensemble complete. Final test prediction shape: {final_test_preds.shape}")

    return final_test_preds