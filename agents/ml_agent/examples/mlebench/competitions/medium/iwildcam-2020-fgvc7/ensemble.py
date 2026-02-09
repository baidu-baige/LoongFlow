import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
import os

# Concrete type definitions for this task
# Predictions: np.ndarray of shape (num_samples, num_classes) containing probabilities
# y: pd.Series containing category_id labels
y = pd.Series
Predictions = np.ndarray

def ensemble(
    all_val_preds: Dict[str, Predictions],
    all_test_preds: Dict[str, Predictions],
    y_val: y,
) -> Predictions:
    """
    Combines predictions from multiple models using soft-voting and performs 
    sequence-level aggregation to ensure temporal consistency across camera trap sequences.
    
    Args:
        all_val_preds: Model name -> Validation probability arrays.
        all_test_preds: Model name -> Test probability arrays.
        y_val: Ground truth validation labels.

    Returns:
        np.ndarray: Final integer class predictions for the test set.
    """
    print("Starting ensemble process: Sequence-level Aggregation...")

    # 1. Determine Validation Mode
    # We must align our metadata loading with the subsetting used in previous stages.
    # The load_data function uses a 200-sample subset if validation_mode is True.
    first_model_preds = next(iter(all_test_preds.values()))
    num_test_samples = len(first_model_preds)
    is_validation_mode = (num_test_samples == 200)

    # 2. Load Metadata for Sequence Information
    # We import load_data locally to avoid circular dependencies if any, 
    # and to retrieve the 'seq_id' required for aggregation.
    from load_data import load_data
    _, _, X_test, test_ids = load_data(validation_mode=is_validation_mode)

    if len(X_test) != num_test_samples:
        raise ValueError(f"Metadata mismatch: X_test has {len(X_test)} samples, but predictions have {num_test_samples}.")

    # 3. Model Weighting (Optimization)
    # If multiple models exist, we could weight them by validation accuracy.
    # For now, we perform simple averaging (Soft Voting).
    print(f"Ensembling {len(all_test_preds)} models...")
    
    test_probs_list = list(all_test_preds.values())
    avg_test_probs = np.mean(test_probs_list, axis=0)

    # 4. Sequence-level Aggregation
    # Animals in the same sequence (seq_id) are almost always the same species.
    # Averaging probabilities across the sequence reduces noise from individual frames.
    print("Applying sequence-level temporal smoothing...")
    
    # Create a DataFrame for efficient grouping
    num_classes = avg_test_probs.shape[1]
    prob_cols = [f'p{i}' for i in range(num_classes)]
    
    # Using a DataFrame with transform('mean') is highly efficient for mapping 
    # sequence-level results back to individual image IDs.
    df_ensemble = pd.DataFrame(avg_test_probs, columns=prob_cols)
    df_ensemble['seq_id'] = X_test['seq_id'].values
    
    # Calculate mean probability per sequence and broadcast back to images
    # This implements the "Aggregation of image-level predictions to sequence-level results"
    seq_avg_probs = df_ensemble.groupby('seq_id')[prob_cols].transform('mean').values

    # 5. Final Prediction Selection
    # Extract the category with the highest aggregated probability.
    final_test_classes = np.argmax(seq_avg_probs, axis=1)

    # 6. Verification and Robustness
    # Ensure no NaN values and that output size is correct.
    if np.isnan(final_test_classes).any():
        print("Warning: NaN detected in ensemble. Filling with class 0.")
        final_test_classes = np.nan_to_num(final_test_classes, nan=0).astype(np.int64)
    
    # Ensure all test samples produce a prediction (default 0 for corrupted images 
    # is already handled by the model predicting on zero-tensors).
    final_test_classes = final_test_classes.astype(np.int64)

    print(f"Ensemble complete. Generated predictions for {len(final_test_classes)} images.")
    
    return final_test_classes