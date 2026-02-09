import numpy as np
import pandas as pd
from typing import Dict, Any
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata

BASE_DATA_PATH = "/mnt/pfs/loongflow/devmachine/1-04/evolux/output/mlebench/histopathologic-cancer-detection/prepared/public"
OUTPUT_DATA_PATH = "output/cf83edc4-8764-4cf8-95a0-4f4a823260c7/2/executor/output"

# Task-adaptive type definitions
y = pd.Series         # Target vector type: Binary labels
Predictions = np.ndarray # Model predictions type: Array of probabilities

def ensemble(
    all_val_preds: Dict[str, Predictions],
    all_test_preds: Dict[str, Predictions],
    y_val: y,
) -> Predictions:
    """
    Combines predictions from multiple models using Weighted Rank Averaging to maximize AUC-ROC.
    
    This implementation handles the aggregation of diverse model architectures (e.g., ConvNeXt and Swin) 
    by converting their probability outputs into ranks, which mitigates scale differences and reduces 
    the impact of outliers or poorly calibrated outputs.
    """
    print("Starting ensemble stage: Weighted Rank Averaging...")

    if not all_test_preds:
        raise ValueError("No test predictions provided for ensemble.")

    model_names = list(all_test_preds.keys())
    num_models = len(model_names)
    test_size = len(next(iter(all_test_preds.values())))

    # 1. Evaluate individual model performance on validation data
    # We use validation AUC to determine weights if multiple models are present.
    model_scores = {}
    weights = {}
    
    print("Evaluating individual model performance (AUC-ROC):")
    for name in model_names:
        if name in all_val_preds:
            # Ensure predictions and ground truth are aligned
            v_preds = all_val_preds[name]
            if len(v_preds) != len(y_val):
                print(f"Warning: Validation prediction length mismatch for {name}. Using equal weight.")
                model_scores[name] = 1.0
            else:
                score = roc_auc_score(y_val, v_preds)
                model_scores[name] = score
                print(f" - {name}: {score:.5f}")
        else:
            print(f"Warning: No validation predictions found for {name}. Using equal weight.")
            model_scores[name] = 1.0

    # 2. Calculate Weights
    # If validation AUCs are available, we use power-weighting to favor stronger models.
    # If only one model exists, weight is 1.0.
    total_score = sum(model_scores.values())
    for name in model_names:
        # Power weighting (e.g., score^2) emphasizes the performance gap
        weights[name] = (model_scores[name] ** 2) / sum([s**2 for s in model_scores.values()])

    # 3. Apply Weighted Rank Averaging on Test Predictions
    # Rank averaging is more robust than simple mean for AUC optimization.
    # Formula: Final_Rank = Sum(Weight_i * Rank_i)
    combined_ranks = np.zeros(test_size, dtype=np.float64)
    
    for name in model_names:
        t_preds = all_test_preds[name]
        
        if np.isnan(t_preds).any():
            raise ValueError(f"NaN values detected in predictions for model: {name}")
            
        # Convert probabilities to ranks (1 to N)
        # Using 'average' method for ties to ensure consistency
        ranks = rankdata(t_preds, method='average')
        
        # Normalize ranks to [0, 1] range to prevent overflow and keep scale consistent
        normalized_ranks = (ranks - 1) / (len(ranks) - 1)
        
        combined_ranks += weights[name] * normalized_ranks
        print(f"Integrated {name} into ensemble with weight: {weights[name]:.4f}")

    # 4. Final Validation
    if np.isnan(combined_ranks).any():
        raise RuntimeError("Ensemble generated NaN values.")

    # Note: Since AUC-ROC only depends on the order of predictions, 
    # the normalized rank sum is a valid substitute for probability.
    print(f"Ensemble complete. Combined {num_models} models.")
    
    return combined_ranks