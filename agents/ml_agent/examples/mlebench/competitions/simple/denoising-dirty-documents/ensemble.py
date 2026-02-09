import numpy as np
from typing import Dict, List, Any

# Task-adaptive type definitions
# Targets and Predictions are 4D numpy arrays: (N, Channels, Height, Width)
y = np.ndarray
Predictions = np.ndarray


def ensemble(
    all_val_preds: Dict[str, Predictions],
    all_test_preds: Dict[str, Predictions],
    y_val: y,
) -> Predictions:
    """
    Combines predictions from multiple models (e.g., across different folds)
    into a final, superior prediction using simple arithmetic averaging.

    Args:
        all_val_preds (Dict[str, Predictions]): Dictionary mapping model/fold names to
                                               their validation (OOF) predictions.
        all_test_preds (Dict[str, Predictions]): Dictionary mapping model/fold names to
                                                their test set predictions.
        y_val (y): Ground truth targets for validation.

    Returns:
        Predictions: Final averaged test set predictions as a 4D numpy array.
    """
    print(f"Starting ensemble process with {len(all_test_preds)} sets of predictions.")

    if not all_test_preds:
        raise ValueError("No test predictions provided for ensembling.")

    # 1. Evaluate individual model scores
    # We calculate RMSE for each model/fold against the provided y_val if shapes match.
    # This provides visibility into the diversity and quality of the ensemble members.
    for name, v_preds in all_val_preds.items():
        if v_preds.shape == y_val.shape:
            mse = np.mean((v_preds - y_val) ** 2)
            rmse = np.sqrt(mse)
            print(f"  [Evaluation] Model: {name} | Val RMSE: {rmse:.6f}")
        else:
            print(
                f"  [Evaluation] Model: {name} | Val shape {v_preds.shape} does not match y_val {y_val.shape}. Skipping RMSE.")

    # 2. Apply ensemble strategy: Simple Arithmetic Mean
    # We aggregate all predictions into a single list and compute the mean across the model dimension.
    test_pred_list = []
    for name, t_preds in all_test_preds.items():
        if np.isnan(t_preds).any() or np.isinf(t_preds).any():
            raise ValueError(f"NaN or Infinity detected in predictions for model: {name}")
        test_pred_list.append(t_preds)

    # Use float64 for intermediate mean calculation to maintain precision
    print(f"  Averaging {len(test_pred_list)} prediction arrays...")
    final_test_preds = np.mean(test_pred_list, axis=0).astype(np.float32)

    # 3. Post-processing and Validation
    # Ensure the final output is bounded correctly within the [0, 1] pixel intensity range.
    final_test_preds = np.clip(final_test_preds, 0.0, 1.0)

    # Final sanity checks
    if np.isnan(final_test_preds).any() or np.isinf(final_test_preds).any():
        raise RuntimeError("Ensemble generated NaN or Infinity values.")

    # Ensure output shape matches input sample count
    expected_samples = next(iter(all_test_preds.values())).shape[0]
    if final_test_preds.shape[0] != expected_samples:
        raise ValueError(f"Ensemble output sample count {final_test_preds.shape[0]} "
                         f"mismatch with expected {expected_samples}")

    print(f"Ensemble complete. Final output shape: {final_test_preds.shape}")

    return final_test_preds