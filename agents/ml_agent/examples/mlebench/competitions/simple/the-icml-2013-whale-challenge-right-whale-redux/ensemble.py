from typing import Dict, List, Any
import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

BASE_DATA_PATH = "/mnt/pfs/loongflow/devmachine/h20-01/evolux/output/mlebench/the-icml-2013-whale-challenge-right-whale-redux/prepared/public"
OUTPUT_DATA_PATH = "output/6795de84-a7ab-443d-bbf5-03771db15966/1/executor/output"

# Task-adaptive type definitions
y = np.ndarray           # Target vector: [N] int64
Predictions = np.ndarray # Model predictions: [N] float32

def ensemble(
    all_val_preds: Dict[str, Predictions],
    all_test_preds: Dict[str, Predictions],
    y_val: y,
) -> Predictions:
    """
    Combines predictions from multiple models using rank averaging to maximize AUC.
    Rank averaging is particularly robust for imbalanced datasets and rank-based metrics like AUC.

    Args:
        all_val_preds (Dict[str, Predictions]): Dictionary mapping model names to OOF predictions.
        all_test_preds (Dict[str, Predictions]): Dictionary mapping model names to test predictions.
        y_val (y): Ground truth targets.

    Returns:
        Predictions: Final test set predictions (rank-averaged scores).
    """
    print(f"Starting ensemble stage. Models provided: {list(all_test_preds.keys())}")

    if not all_test_preds:
        raise ValueError("The 'all_test_preds' dictionary is empty. No predictions to ensemble.")

    # Step 1: Evaluate individual model performance
    # This provides visibility into which models/folds are performing best.
    for model_name, val_p in all_val_preds.items():
        try:
            if len(val_p) == len(y_val):
                auc = roc_auc_score(y_val, val_p)
                print(f"Model [{model_name}] Out-of-Fold AUC: {auc:.5f}")
            else:
                # If y_val doesn't match the OOF size (e.g., if it's only one fold), log and skip.
                print(f"Model [{model_name}] AUC skip: y_val size {len(y_val)} != OOF size {len(val_p)}")
        except Exception as e:
            print(f"Model [{model_name}] AUC evaluation failed: {e}")

    # Step 2: Apply Rank-based Aggregation
    # Strategy: Convert each model's predictions into ranks, then average those ranks.
    # This prevents models with different probability calibrations from dominating the ensemble.
    print(f"Applying Rank Averaging on {len(all_test_preds)} models...")
    
    all_percentiles = []
    # Sort keys for deterministic behavior
    model_keys = sorted(all_test_preds.keys())
    
    for name in model_keys:
        t_preds = all_test_preds[name]
        
        # Calculate ranks using scipy.stats.rankdata with 'average' tie-breaking as specified.
        # This converts raw probabilities into a ranking from 1 to N.
        ranks = rankdata(t_preds, method='average')
        
        # Normalize to percentile range (0, 1] to ensure all models have equitable weight
        # regardless of the number of samples or tie distribution.
        percentiles = ranks / float(len(t_preds))
        all_percentiles.append(percentiles)
    
    # Simple mean of the percentile ranks
    final_test_scores = np.mean(all_percentiles, axis=0)

    # Step 3: Final validation
    if not np.isfinite(final_test_scores).all():
        raise RuntimeError("Ensemble generated non-finite values (NaN or Infinity).")

    print(f"Ensemble completed. Output length: {len(final_test_scores)}")
    
    return final_test_scores.astype(np.float32)