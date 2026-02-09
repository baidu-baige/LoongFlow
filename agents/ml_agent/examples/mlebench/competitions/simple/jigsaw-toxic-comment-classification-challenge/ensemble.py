import numpy as np
import pandas as pd
from typing import Dict, Any
from sklearn.metrics import roc_auc_score

BASE_DATA_PATH = "/mnt/pfs/loongflow/devmachine/h20-03/evolux/output/mlebench/jigsaw-toxic-comment-classification-challenge/prepared/public"
OUTPUT_DATA_PATH = "output/4d08636e-bf37-40e0-b9d7-8ffb77d57ea2/1/executor/output"

# Task-adaptive type definitions
y = np.ndarray           # Target vector type: Multi-label binary targets
Predictions = np.ndarray # Model predictions type: Probabilities for 6 classes

def ensemble(
    all_val_preds: Dict[str, Predictions],
    all_test_preds: Dict[str, Predictions],
    y_val: y,
) -> Predictions:
    """
    Combines predictions from multiple folds/models using simple arithmetic averaging.

    Args:
        all_val_preds (Dict[str, Predictions]): Dictionary mapping model/fold names to OOF predictions.
        all_test_preds (Dict[str, Predictions]): Dictionary mapping model/fold names to test set predictions.
        y_val (y): Ground truth targets for the validation set.

    Returns:
        Predictions: Final averaged test set predictions of shape (test_size, 6).
    """
    print(f"Starting ensemble process with {len(all_test_preds)} sets of predictions.")

    # Step 1: Evaluate individual model scores
    # This helps monitor if any specific fold/model is performing poorly.
    target_columns = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    
    for name, val_preds in all_val_preds.items():
        try:
            # Calculate column-wise ROC AUC and then the mean
            # Ensure indices match for metric calculation
            if val_preds.shape == y_val.shape:
                individual_aucs = []
                for i in range(y_val.shape[1]):
                    auc = roc_auc_score(y_val[:, i], val_preds[:, i])
                    individual_aucs.append(auc)
                
                mean_auc = np.mean(individual_aucs)
                print(f"Model/Fold '{name}' Validation Mean ROC AUC: {mean_auc:.6f}")
            else:
                print(f"Model/Fold '{name}' validation shape mismatch. Skipping OOF evaluation.")
        except Exception as e:
            print(f"Could not calculate score for {name}: {str(e)}")

    # Step 2: Apply ensemble strategy - Simple Arithmetic Mean
    # The technical specification requires averaging across all folds/models.
    test_preds_list = list(all_test_preds.values())
    
    if not test_preds_list:
        raise ValueError("No test predictions found in all_test_preds.")

    # Stack and calculate mean across the first axis (the list axis)
    # Using float64 for the intermediate summation to maintain precision
    all_test_stacked = np.array(test_preds_list, dtype=np.float64)
    final_test_preds = np.mean(all_test_stacked, axis=0).astype(np.float32)

    # Step 3: Sanity and Quality Checks
    if np.isnan(final_test_preds).any():
        raise ValueError("Ensemble result contains NaN values.")
    
    if np.isinf(final_test_preds).any():
        raise ValueError("Ensemble result contains Infinity values.")
    
    # Ensure probabilities are clipped between 0 and 1
    final_test_preds = np.clip(final_test_preds, 0.0, 1.0)

    # Validate output shape against the input predictions
    expected_shape = test_preds_list[0].shape
    if final_test_preds.shape != expected_shape:
        raise ValueError(f"Shape mismatch: Expected {expected_shape}, got {final_test_preds.shape}")

    print(f"Ensemble completed successfully. Final prediction shape: {final_test_preds.shape}")
    
    return final_test_preds