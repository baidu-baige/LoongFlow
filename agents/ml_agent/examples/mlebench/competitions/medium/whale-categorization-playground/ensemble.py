import os
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from typing import Dict, Any, List

# Target vector type (LongTensor from upstream or np.ndarray)
y = Any 
# Model predictions type (np.ndarray of scores or final ID strings)
Predictions = Any 

def ensemble(
    all_val_preds: Dict[str, Predictions],
    all_test_preds: Dict[str, Predictions],
    y_val: y,
) -> Predictions:
    """
    Combines predictions from multiple models into a final robust output.
    Aggregates multi-fold/multi-model signals and optimizes 'new_whale' injection threshold.
    """
    print("Ensemble stage starting...")

    # Step 1: Combine Model Predictions
    def combine_preds(preds_dict: Dict[str, np.ndarray]) -> np.ndarray:
        if not preds_dict:
            raise ValueError("Prediction dictionary is empty.")
        
        vals = list(preds_dict.values())
        first_shape = vals[0].shape
        
        # If all values have the same shape, they represent different models on the same samples
        # If they have different shapes, they represent OOF folds — concatenate them
        all_same_shape = all(v.shape == first_shape for v in vals)
        
        if all_same_shape:
            print(f"Averaging {len(vals)} models with shape {first_shape}")
            return np.mean(vals, axis=0)
        else:
            print(f"Concatenating {len(vals)} folds for OOF reconstruction")
            # Sorting keys ensures potential alignment with y_val if it was concatenated
            sorted_keys = sorted(preds_dict.keys())
            return np.concatenate([preds_dict[k] for k in sorted_keys], axis=0)

    avg_val_logits = combine_preds(all_val_preds)
    avg_test_logits = combine_preds(all_test_preds)

    # Convert logits to probabilities using Softmax (stable via torch)
    def get_probs(logits: np.ndarray) -> np.ndarray:
        t_logits = torch.from_numpy(logits)
        return torch.softmax(t_logits, dim=1).numpy()

    val_probs = get_probs(avg_val_logits)
    test_probs = get_probs(avg_test_logits)
    num_classes = val_probs.shape[1]

    # Step 2: Reconstruct Label Mapping
    # Logic: Match the LabelEncoder fit in preprocess.py
    BASE_DATA_PATH = "/mnt/pfs/loongflow/devmachine/2-05/evolux/output/mlebench/whale-categorization-playground/prepared/public"
    train_csv_path = os.path.join(BASE_DATA_PATH, "train.csv")
    train_df = pd.read_csv(train_csv_path)
    
    # Validation mode check: load_data subsets to 200 rows
    if num_classes < 4029:
        print("Validation mode detected based on class count. Subsetting training metadata...")
        train_df = train_df.head(200)
    
    le = LabelEncoder()
    le.fit(train_df['Id'])
    
    if len(le.classes_) != num_classes:
        raise RuntimeError(f"Class count mismatch: Model has {num_classes}, LabelEncoder has {len(le.classes_)}")

    idx_new_whale = le.transform(['new_whale'])[0]
    known_indices = np.array([i for i in range(num_classes) if i != idx_new_whale])

    # Step 3: Threshold Optimization on OOF
    # Ensure y_val is aligned numpy array
    if torch.is_tensor(y_val):
        y_val_np = y_val.cpu().numpy()
    else:
        y_val_np = np.array(y_val)

    # Pre-calculate components for grid search speed
    val_probs_known = val_probs[:, known_indices]
    val_max_known = np.max(val_probs_known, axis=1)
    val_top4_known_idx = known_indices[np.argsort(val_probs_known, axis=1)[:, -4:][:, ::-1]]
    val_top5_all = np.argsort(val_probs, axis=1)[:, -5:][:, ::-1]

    def map5_score(labels: np.ndarray, preds: np.ndarray) -> float:
        hits = (preds == labels[:, None])
        weights = np.array([1.0, 0.5, 0.33333333, 0.25, 0.2])
        return np.sum(hits * weights) / len(labels)

    print("Optimizing 'new_whale' threshold on OOF predictions...")
    best_threshold = 0.5
    best_score = -1.0
    
    # Grid search threshold from 0.1 to 0.9
    for t in np.linspace(0.1, 0.9, 81):
        mask = val_max_known < t
        tmp_preds = np.zeros((len(y_val_np), 5), dtype=int)
        
        # Strategy: If max_prob < threshold, start with 'new_whale'
        tmp_preds[mask, 0] = idx_new_whale
        tmp_preds[mask, 1:] = val_top4_known_idx[mask]
        # Otherwise, standard top 5
        tmp_preds[~mask] = val_top5_all[~mask]
        
        score = map5_score(y_val_np, tmp_preds)
        if score > best_score:
            best_score = score
            best_threshold = t

    print(f"Optimization complete. Best Threshold: {best_threshold:.4f}, Validation MAP@5: {best_score:.4f}")

    # Step 4: Generate Final Test Predictions
    num_test = len(test_probs)
    test_probs_known = test_probs[:, known_indices]
    test_max_known = np.max(test_probs_known, axis=1)
    test_top4_known_idx = known_indices[np.argsort(test_probs_known, axis=1)[:, -4:][:, ::-1]]
    test_top5_all = np.argsort(test_probs, axis=1)[:, -5:][:, ::-1]

    final_test_indices = np.zeros((num_test, 5), dtype=int)
    mask_test = test_max_known < best_threshold
    
    final_test_indices[mask_test, 0] = idx_new_whale
    final_test_indices[mask_test, 1:] = test_top4_known_idx[mask_test]
    final_test_indices[~mask_test] = test_top5_all[~mask_test]

    # Convert indices back to whale IDs
    print("Mapping indices to whale IDs...")
    flat_indices = final_test_indices.ravel()
    flat_ids = le.inverse_transform(flat_indices)
    test_id_matrix = flat_ids.reshape(num_test, 5)
    
    # Format as space-separated strings as per submission requirement
    final_predictions = np.array([" ".join(row) for row in test_id_matrix])

    print(f"Ensemble complete. Generated {len(final_predictions)} predictions.")
    return final_predictions