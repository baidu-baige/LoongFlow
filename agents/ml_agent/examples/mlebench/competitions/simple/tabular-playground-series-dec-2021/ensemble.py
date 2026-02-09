from typing import Dict, List, Any
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

BASE_DATA_PATH = "/mnt/pfs/loongflow/devmachine/h20-04/evolux/output/mlebench/tabular-playground-series-dec-2021/prepared/public"
OUTPUT_DATA_PATH = "output/477e9955-ebee-46f4-96b0-878df6f022f5/1/executor/output"

# Task-adaptive type definitions
# y is the target series, Predictions type represents model probability outputs (arrays) 
# or final class labels (arrays).
y = pd.Series
Predictions = np.ndarray

def ensemble(
    all_val_preds: Dict[str, Predictions],
    all_test_preds: Dict[str, Predictions],
    y_val: y,
) -> Predictions:
    """
    Combines probabilistic predictions from multiple folds into a final class output using Soft Voting.
    
    Args:
        all_val_preds (Dict[str, Predictions]): Dictionary mapping model/fold names to out-of-fold probabilities.
        all_test_preds (Dict[str, Predictions]): Dictionary mapping model/fold names to test set probabilities.
        y_val (y): Ground truth targets for validation.

    Returns:
        Predictions: Final test set class predictions (integers).
    """
    # Technical Specification: The target classes are [1, 2, 3, 4, 6, 7] (5 was removed).
    # Since XGBoost was trained on LabelEncoded targets, the probability columns follow 
    # the sorted order of these classes.
    classes = np.array([1, 2, 3, 4, 6, 7], dtype=np.int32)
    
    print(f"Ensemble Stage: Combining results from {len(all_test_preds)} folds/models.")

    # Step 1: Evaluate individual model scores and prediction correlations
    model_names = list(all_test_preds.keys())
    
    # 1.1: Out-of-Fold (OOF) Scoring
    # If the provided y_val matches the total OOF size, we calculate global accuracy.
    # Otherwise, we attempt to score individual entries if they align.
    for name, v_probs in all_val_preds.items():
        if v_probs.shape[0] == len(y_val):
            v_labels = classes[np.argmax(v_probs, axis=1)]
            score = accuracy_score(y_val, v_labels)
            print(f"Validation Accuracy for {name}: {score:.6f}")
        else:
            # Predictions might be fold-specific slices
            pass

    # 1.2: Test Prediction Correlation Analysis
    if len(model_names) > 1:
        print("Analyzing correlations between model test predictions...")
        # We use a small subset of models for correlation logging to avoid excessive output
        sample_models = model_names[:min(5, len(model_names))]
        for i in range(len(sample_models)):
            for j in range(i + 1, len(sample_models)):
                m1, m2 = sample_models[i], sample_models[j]
                # Correlate based on argmax class indices
                p1_idx = np.argmax(all_test_preds[m1], axis=1)
                p2_idx = np.argmax(all_test_preds[m2], axis=1)
                corr = np.corrcoef(p1_idx, p2_idx)[0, 1]
                print(f"Correlation ({m1}, {m2}): {corr:.4f}")

    # Step 2: Apply ensemble strategy - Soft Voting (Probability Averaging)
    # Objective: Stabilize variance across folds by averaging predicted class probabilities.
    print("Executing Soft Voting (Mean Probability Aggregation)...")
    
    # Efficiently stack all test probability arrays
    # Each array is (n_samples, n_classes)
    test_probs_list = [all_test_preds[name] for name in model_names]
    
    # Utilize numpy for parallelized mean calculation across the fold/model axis (axis=0)
    # The result has shape (n_samples, n_classes)
    avg_test_probs = np.mean(np.stack(test_probs_list, axis=0), axis=0)
    
    # Step 3: Generate final test predictions
    # Select the class index with the highest average probability
    final_indices = np.argmax(avg_test_probs, axis=1)
    
    # Map back to original Forest Cover Type integers
    final_test_preds = classes[final_indices]
    
    # Final Validation & Quality Control
    if np.isnan(final_test_preds).any():
        raise ValueError("Ensemble process produced NaN values in final predictions.")
    
    # Check consistency with test set size
    expected_size = len(test_probs_list[0])
    if len(final_test_preds) != expected_size:
        raise ValueError(f"Ensemble output size ({len(final_test_preds)}) does not match test size ({expected_size}).")

    print("Ensemble complete. Generated predictions for test set.")
    # Log distribution to verify minority class representation (like Class 4)
    pred_dist = pd.Series(final_test_preds).value_counts().sort_index().to_dict()
    print(f"Final Class Distribution: {pred_dist}")

    return final_test_preds