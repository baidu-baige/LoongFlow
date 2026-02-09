import numpy as np
import pandas as pd
from typing import Dict, Any
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

BASE_DATA_PATH = "/mnt/pfs/loongflow/devmachine/2-05/evolux/output/mlebench/random-acts-of-pizza/prepared/public"
OUTPUT_DATA_PATH = "output/f2dbb22d-a0cb-4add-aa87-f2c6b1a4b76f/77/executor/output"

# Task-adaptive type definitions for the Random Acts of Pizza challenge
y = pd.Series           # Target vector type (requester_received_pizza)
Predictions = np.ndarray # Probability array [N,] representing success likelihood

def ensemble(
    all_val_preds: Dict[str, Predictions],
    all_test_preds: Dict[str, Predictions],
    y_val: y,
) -> Predictions:
    """
    Combines predictions from multiple models into a final output using Rank-Weighted Averaging (RWA).
    
    This implementation leverages egalitarian weighting (0.2 each) across five diverse specialists:
    LGBM, CatBoost, XGBoost, Linear (Logistic Regression), and Transformer (DeBERTa).
    Rank-transformation mitigates calibration variance and scale differences across heterogeneous models.
    """
    print("Ensemble Stage: Initializing egalitarian Rank-Weighted Averaging (RWA)...")

    # Step 1: Define Target Weights from Technical Specification
    # These represent the egalitarian contribution of 3x GBDT, 1x Linear, and 1x Transformer.
    target_weights = {
        'lgbm': 0.2,
        'catboost': 0.2,
        'xgboost': 0.2,
        'lr': 0.2,
        'deberta': 0.2
    }

    # Step 2: Identify available models in the current pipeline context
    # We check for keys containing the target names to be robust to naming conventions (e.g., 'lgbm_specialist')
    models_to_use = []
    use_weights = {}
    
    for target in target_weights.keys():
        match = [k for k in all_val_preds.keys() if target in k.lower()]
        if match:
            models_to_use.append(match[0])
            use_weights[match[0]] = target_weights[target]
    
    # Fallback/Safety Check: If specific keys are missing, we use all available engines
    if not models_to_use:
        print("Warning: Specific target keys not found. Aggregating all available prediction engines.")
        models_to_use = list(all_val_preds.keys())
        if not models_to_use:
            raise KeyError("Critical Error: No models found in all_val_preds dictionary.")
        # Re-distribute weights equally
        weight_val = 1.0 / len(models_to_use)
        use_weights = {m: weight_val for m in models_to_use}
    else:
        # Re-normalize weights if only a subset of the 5 specialists is present
        total_weight = sum(use_weights.values())
        for m in use_weights:
            use_weights[m] /= total_weight

    print(f"Selected models for ensemble: {list(use_weights.keys())}")
    print(f"Applied weights: {use_weights}")

    # Step 3: Performance Audit and Correlation Analysis
    print("Individual Model Validation Performance (ROC-AUC):")
    for name in models_to_use:
        score = roc_auc_score(y_val, all_val_preds[name])
        print(f"  - {name}: {score:.4f}")

    if len(models_to_use) > 1:
        print("Inter-model Prediction Correlation Matrix:")
        corr_matrix = pd.DataFrame({m: all_val_preds[m] for m in models_to_use}).corr()
        print(corr_matrix)

    # Step 4: Implement Rank-Transformation Helper
    def get_rank_normalized_preds(preds: Predictions) -> Predictions:
        """
        Converts raw probabilities to normalized ranks in the [0, 1] range.
        Uses the 'average' method to handle ties consistently across samples.
        """
        if len(preds) <= 1:
            return preds
        # rankdata returns ranks starting from 1
        ranks = rankdata(preds, method='average')
        # Scale to [0, 1] using (rank - 1) / (N - 1)
        return (ranks - 1) / (len(ranks) - 1)

    # Step 5: Compute Ensemble Validation Score (OOF Estimate)
    final_val_preds = np.zeros(len(y_val), dtype=float)
    for name in models_to_use:
        # Validate data integrity
        if not np.isfinite(all_val_preds[name]).all():
            raise ValueError(f"Model '{name}' contains non-finite values in validation predictions.")
        
        norm_ranks = get_rank_normalized_preds(all_val_preds[name])
        final_val_preds += use_weights[name] * norm_ranks

    ensemble_val_score = roc_auc_score(y_val, final_val_preds)
    print(f"Ensemble Validation ROC-AUC: {ensemble_val_score:.4f}")

    # Step 6: Generate Final Test Predictions
    # Identical rank-transformation and weighting logic must be applied to the test set
    test_sample_count = len(next(iter(all_test_preds.values())))
    final_test_preds = np.zeros(test_sample_count, dtype=float)

    for name in models_to_use:
        if name not in all_test_preds:
            raise KeyError(f"Critical Error: Model '{name}' found in validation set but missing in test set.")
            
        test_preds = all_test_preds[name]
        if not np.isfinite(test_preds).all():
            raise ValueError(f"Model '{name}' contains non-finite values in test predictions.")
        
        norm_ranks = get_rank_normalized_preds(test_preds)
        final_test_preds += use_weights[name] * norm_ranks

    # Step 7: Final Integrity Checks
    if not np.isfinite(final_test_preds).all():
        raise ValueError("Critical Error: Ensemble generated non-finite values.")
    
    if len(final_test_preds) != test_sample_count:
        raise ValueError(f"Output mismatch: expected {test_sample_count} samples, got {len(final_test_preds)}")

    print("Ensemble stage completed successfully.")
    return final_test_preds