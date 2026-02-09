import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from typing import Dict, Any

# Task-adaptive type definitions
y = pd.Series  # Target vector type (binary labels)
Predictions = np.ndarray  # Model predictions type (1D array of probabilities)


def ensemble(
    all_val_preds: Dict[str, Predictions],
    all_test_preds: Dict[str, Predictions],
    y_val: y,
) -> Predictions:
    """
    Combines predictions from multiple models using Weighted Rank Averaging.

    This method maximizes AUC by focusing on the relative ranking of samples rather
    than absolute probability values, effectively consolidating semantic (DeBERTa)
    and statistical (TF-IDF+Ridge) signals.
    """
    print("Stage 5: Starting Weighted Rank Averaging Ensemble...")

    # Step 1: Identify models and assign weights based on Technical Specification
    model_names = list(all_test_preds.keys())
    if not model_names:
        raise RuntimeError("Ensemble failed: No model predictions found in all_test_preds.")

    weights = {}
    for name in model_names:
        lname = name.lower()
        # Weight DeBERTa: 0.7, Weight TF-IDF+Ridge: 0.3
        if "deberta" in lname and "ridge" not in lname:
            weights[name] = 0.7
        elif "ridge" in lname and "deberta" not in lname:
            weights[name] = 0.3
        else:
            # Handle combined models or unknown models with unit weight
            # If 'deberta_ridge_ensemble' is the only key, it gets 1.0
            weights[name] = 1.0

    # Step 2: Evaluate individual model performance on validation set
    print("Evaluating individual model performance (AUC)...")
    for name in model_names:
        if name in all_val_preds:
            score = roc_auc_score(y_val, all_val_preds[name])
            print(f"Model [{name}] (Weight: {weights[name]:.2f}) Validation AUC: {score:.4f}")

    # Step 3: Weighted Rank Averaging
    # We apply the ranking to both Val (for monitoring) and Test (for final output)
    def calculate_rank_average(preds_dict: Dict[str, Predictions], weight_map: Dict[str, float]) -> np.ndarray:
        first_key = next(iter(preds_dict))
        num_samples = len(preds_dict[first_key])
        weighted_ranks = np.zeros(num_samples, dtype=np.float64)
        sum_weights = 0.0

        for name, probs in preds_dict.items():
            w = weight_map.get(name, 1.0)
            # Convert raw probabilities to ranks. method='average' handles ties.
            ranks = rankdata(probs, method='average')
            weighted_ranks += w * ranks
            sum_weights += w

        # Normalize the weighted rank sum
        # Scale back to [0, 1] range to satisfy probabiltiy-like output requirement
        r_min, r_max = weighted_ranks.min(), weighted_ranks.max()
        if r_max > r_min:
            final_probs = (weighted_ranks - r_min) / (r_max - r_min)
        else:
            # Fallback for constant predictions
            final_probs = np.zeros_like(weighted_ranks)

        return final_probs

    # Calculate validation ensemble for progress tracking
    val_ensemble_probs = calculate_rank_average(all_val_preds, weights)
    ensemble_val_auc = roc_auc_score(y_val, val_ensemble_probs)
    print(f"Ensemble Validation AUC: {ensemble_val_auc:.4f}")

    # Calculate final test predictions
    print(f"Generating ensembled test predictions for {len(model_names)} models...")
    final_test_preds = calculate_rank_average(all_test_preds, weights)

    # Step 4: Final Integrity Checks
    # Ensure no NaN or Inf values
    if np.isnan(final_test_preds).any() or np.isinf(final_test_preds).any():
        final_test_preds = np.nan_to_num(final_test_preds, nan=0.0, posinf=1.0, neginf=0.0)

    # Ensure output matches target test size
    expected_size = len(next(iter(all_test_preds.values())))
    if len(final_test_preds) != expected_size:
        raise RuntimeError(f"Alignment Error: Ensemble size {len(final_test_preds)} != Expected {expected_size}")

    print("Ensemble stage complete.")
    return final_test_preds.astype(np.float32)