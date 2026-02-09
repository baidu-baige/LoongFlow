import os
import joblib
import numpy as np
import torch
from typing import Dict, List, Any
from sklearn.metrics import f1_score
from scipy import stats

# Pipeline configuration constants
BASE_DATA_PATH = "/mnt/pfs/loongflow/devmachine/h20-02/evolux/output/mlebench/herbarium-2020-fgvc7/prepared/public"
OUTPUT_DATA_PATH = "output/cf84c6d3-8647-45ca-b9ff-02e7ed67cf5b/1/executor/output"

# Task-adaptive type definitions
# y: Target vector containing ground truth class indices (local mapping 0..N-1)
# Predictions: Model output containing either class probabilities or class indices
y = Any           
Predictions = np.ndarray 

def ensemble(
    all_val_preds: Dict[str, Predictions],
    all_test_preds: Dict[str, Predictions],
    y_val: y,
) -> Predictions:
    """
    Aggregates model outputs for maximum robustness across species.
    
    This implementation:
    1. Loads the category mapping to understand the local vs. global label space.
    2. Handles the discrepancy between model output dimension (32,093) and subset class count.
    3. Performs majority voting or mean probability aggregation.
    4. Maps local indices back to the original COCO category_ids.
    """
    print(f"Ensemble stage started with {len(all_test_preds)} prediction sets.")

    # Step 1: Load the category mapping saved during preprocessing
    mapping_path = os.path.join(OUTPUT_DATA_PATH, "category_mapping.joblib")
    if not os.path.exists(mapping_path):
        raise FileNotFoundError(f"Category mapping not found at {mapping_path}.")
    
    mapping_data = joblib.load(mapping_path)
    idx_to_cat = mapping_data['idx_to_cat']
    num_classes = len(idx_to_cat)
    print(f"Detected {num_classes} unique species in the current validation/training subset.")

    # Step 2: Evaluate individual model performance on validation set
    if torch.is_tensor(y_val):
        y_true = y_val.cpu().numpy()
    else:
        y_true = np.asarray(y_val)

    print("--- Individual Model Performance (Validation Macro F1) ---")
    for model_name, val_preds in all_val_preds.items():
        # Handle cases where model predicts global indices (up to 32092) 
        # while targets are local (0..num_classes-1)
        if val_preds.ndim == 2:
            # If probabilities are provided, restrict argmax to the known class range
            v_idx = np.argmax(val_preds[:, :num_classes], axis=1)
        else:
            # If hard indices are provided, clip them to the valid local range
            v_idx = np.clip(val_preds, 0, num_classes - 1)
        
        score = f1_score(y_true, v_idx, average='macro')
        print(f" - {model_name}: {score:.5f}")

    # Step 3: Aggregate test predictions
    model_keys = list(all_test_preds.keys())
    if not model_keys:
        raise ValueError("No test predictions provided for ensemble.")
        
    first_pred = all_test_preds[model_keys[0]]
    num_samples = len(first_pred)

    # Strategy: Mean aggregation for probabilities, Mode aggregation (voting) for indices
    if first_pred.ndim == 2:
        print(f"Aggregating probability matrices (mean aggregation)...")
        total_probs = np.zeros((num_samples, num_classes), dtype=np.float64)
        for name in model_keys:
            # Restrict to num_classes to handle discrepancy between global head and local mapping
            probs = all_test_preds[name][:, :num_classes]
            total_probs += probs
        avg_probs = total_probs / len(model_keys)
        final_indices = np.argmax(avg_probs, axis=1)
    else:
        print(f"Aggregating hard labels (majority voting)...")
        # Ensure all predictions are clipped to the valid mapping range [0, num_classes-1]
        # This prevents IndexError when mapping to category_id
        processed_preds = []
        for name in model_keys:
            clipped = np.clip(all_test_preds[name], 0, num_classes - 1)
            processed_preds.append(clipped)
        
        preds_stack = np.stack(processed_preds)
        # scipy.stats.mode returns (mode_values, counts)
        mode_res = stats.mode(preds_stack, axis=0)
        final_indices = np.asarray(mode_res.mode).reshape(-1)

    # Step 4: Map local indices back to original COCO category_ids
    print("Mapping final indices to original category IDs...")
    # Extract keys and values for building a robust lookup table
    # Mapping might not be 0..N-1 if something went wrong in preprocess, so we handle gaps
    max_idx = max(idx_to_cat.keys())
    lookup_table = np.zeros(max_idx + 1, dtype=np.int64)
    for idx, cat_id in idx_to_cat.items():
        lookup_table[idx] = cat_id
    
    # Final safety clip before indexing
    final_indices = np.clip(final_indices, 0, max_idx)
    final_test_predictions = lookup_table[final_indices]

    # Step 5: Final Consistency Check
    if len(final_test_predictions) != num_samples:
        raise ValueError(f"Output count mismatch: got {len(final_test_predictions)}, expected {num_samples}")
    
    print(f"Ensemble complete. Generated predictions for {num_samples} test images.")
    return final_test_predictions