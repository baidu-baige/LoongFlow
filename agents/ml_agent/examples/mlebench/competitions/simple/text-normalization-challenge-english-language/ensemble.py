import numpy as np
from typing import Dict, Any

# Task-adaptive type definitions for English Text Normalization
y = np.ndarray           # Ground truth 'after' tokens (strings)
Predictions = np.ndarray # Predicted 'after' tokens (strings)

def ensemble(
    all_val_preds: Dict[str, Predictions],
    all_test_preds: Dict[str, Predictions],
    y_val: y,
) -> Predictions:
    """
    Combines predictions from multiple models into a final superior output.
    
    This implementation follows a Hierarchical Fallback Strategy at the ensemble level:
    1. Evaluate each model's precision on validation data.
    2. Use the most frequent prediction across models (Majority Voting).
    3. Resolve ties by prioritizing the prediction from the model with the highest 
       validation accuracy (Deterministic Preference).
    
    This ensures that high-precision rule-based mappings (which tend to be consistent 
    across models) are preserved, while generalized model predictions are combined 
    robustly for exact match accuracy.
    """
    if not all_test_preds:
        raise ValueError("Ensemble received an empty collection of test predictions.")

    print(f"Initializing ensemble for {len(all_test_preds)} models...")

    # Step 1: Evaluate individual model scores
    model_accuracies = {}
    for name, preds in all_val_preds.items():
        # Evaluation is based on Exact Match Accuracy (as per competition rules)
        score = np.mean(preds == y_val)
        model_accuracies[name] = score
        print(f"Model '{name}' Out-of-Fold Accuracy: {score:.6f}")

    # Identify the best-performing model to serve as the tie-breaker
    sorted_models = sorted(model_accuracies.keys(), key=lambda x: model_accuracies[x], reverse=True)
    best_model_name = sorted_models[0]
    print(f"Primary model for tie-breaking: {best_model_name} (Acc: {model_accuracies[best_model_name]:.6f})")

    # Optimization: If only one model exists, return its predictions immediately
    if len(all_test_preds) == 1:
        print("Single model ensemble: Returning predictions from the primary engine.")
        return all_test_preds[best_model_name]

    # Step 2: Majority Vote with Best-Model Tie-breaking
    print("Executing majority voting strategy...")
    
    # Establish consistent model order: Best model at index 0
    ordered_names = [best_model_name] + [m for m in all_test_preds.keys() if m != best_model_name]
    
    # Stack predictions into a 2D matrix for vectorized comparisons
    # Shape: (num_models, num_samples)
    test_preds_matrix = np.array([all_test_preds[name] for name in ordered_names], dtype=object)
    num_models, num_samples = test_preds_matrix.shape
    
    # Initialize final output with the best model's predictions
    final_preds = test_preds_matrix[0].copy()
    
    # Identify indices where models disagree to minimize redundant computation
    # We check if every model's prediction equals the best model's prediction
    disagreement_mask = (test_preds_matrix != test_preds_matrix[0]).any(axis=0)
    diff_indices = np.where(disagreement_mask)[0]
    
    print(f"Disagreement detected in {len(diff_indices)} / {num_samples} tokens.")

    if len(diff_indices) > 0:
        # Resolve disagreements sample-by-sample
        for idx in diff_indices:
            votes = test_preds_matrix[:, idx]
            
            # Efficiently count occurrences of each string prediction
            counts = {}
            for v in votes:
                counts[v] = counts.get(v, 0) + 1
            
            # Find the maximum number of votes
            max_votes = max(counts.values())
            
            # Identify all candidates with the maximum vote count
            winners = [v for v, c in counts.items() if c == max_votes]
            
            if len(winners) == 1:
                # Unambiguous majority
                final_preds[idx] = winners[0]
            else:
                # Tie: Hierarchical fallback to the highest-scoring individual model
                # The best_model prediction is at votes[0]
                best_pred = votes[0]
                if best_pred in winners:
                    final_preds[idx] = best_pred
                else:
                    # In the rare case the best model isn't among the winners, 
                    # use the first winner found (stable choice)
                    final_preds[idx] = winners[0]

    # Final sanity check: Verify ensemble performance on validation data
    val_preds_matrix = np.array([all_val_preds[name] for name in ordered_names], dtype=object)
    val_ensemble = val_preds_matrix[0].copy()
    val_diff_mask = (val_preds_matrix != val_preds_matrix[0]).any(axis=0)
    val_diff_indices = np.where(val_diff_mask)[0]
    
    for idx in val_diff_indices:
        v_votes = val_preds_matrix[:, idx]
        v_counts = {}
        for v in v_votes: v_counts[v] = v_counts.get(v, 0) + 1
        v_max = max(v_counts.values())
        v_winners = [v for v, c in v_counts.items() if c == v_max]
        val_ensemble[idx] = v_votes[0] if v_votes[0] in v_winners else v_winners[0]
        
    ens_acc = np.mean(val_ensemble == y_val)
    print(f"Ensemble Validation Accuracy: {ens_acc:.6f} (Gain over best single model: {ens_acc - model_accuracies[best_model_name]:.6f})")

    print("Ensemble stage completed.")
    return final_preds