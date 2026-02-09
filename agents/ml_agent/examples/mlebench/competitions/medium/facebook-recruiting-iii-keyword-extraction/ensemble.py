import pandas as pd
import numpy as np
import scipy.sparse as sp
import os
import pickle
import gc
from typing import Dict, List, Any
from collections import Counter

BASE_DATA_PATH = "/mnt/pfs/loongflow/devmachine/h20-03/evolux/output/mlebench/facebook-recruiting-iii-keyword-extraction/prepared/public"
OUTPUT_DATA_PATH = "output/90689814-166b-4e4f-a971-355572d18239/1/executor/output"

# Task-adaptive type definitions
# y: Sparse binary matrix for multi-label targets
# Predictions: List of space-delimited tag strings for the competition format
y = sp.csr_matrix
Predictions = List[str]

def ensemble(
    all_val_preds: Dict[str, np.ndarray],
    all_test_preds: Dict[str, np.ndarray],
    y_val: y,
) -> Predictions:
    """
    Combines predictions from multiple models into a final output string for the test set.
    
    Strategy:
    1. Average probability estimates from all models.
    2. Reconstruct the tag vocabulary used during preprocessing.
    3. Generate Top-K predictions (up to 3 tags) based on a confidence threshold (0.2).
    4. Override model predictions with known tags from the training set (duplicate lookup) for maximum accuracy.
    """
    print("Stage 1/4: Analyzing input dimensions and execution mode...")
    # Determine sizing from inputs
    # We assume at least one model exists in the dictionary (e.g., 'sgd_ovr')
    first_model_test = next(iter(all_test_preds.values()))
    n_test = first_model_test.shape[0]
    num_classes = first_model_test.shape[1]
    n_val = y_val.shape[0]
    
    # Mode detection to ensure consistency with upstream preprocessing
    # In validation mode, load_data uses 200 rows, so y_val (10%) has ~20 rows.
    is_val_mode = n_val < 1000
    n_train_to_read = 200 if is_val_mode else None
    
    print(f"Detected mode: {'Validation' if is_val_mode else 'Full'}")
    print(f"Test samples: {n_test}, Target classes: {num_classes}")

    # Stage 2: Reconstruct Tag Vocabulary
    # The columns of the prediction matrix correspond to the top-N tags from the training set.
    print("Stage 2/4: Reconstructing tag vocabulary from training data...")
    train_csv = os.path.join(BASE_DATA_PATH, "train.csv")
    
    # Read only necessary columns and rows to manage memory
    train_df = pd.read_csv(train_csv, usecols=['Title', 'Body', 'Tags'], nrows=n_train_to_read, low_memory=True)
    # Match the deduplication logic from load_data.py to ensure tag counts are consistent
    train_df = train_df.drop_duplicates(subset=['Title', 'Body'])
    
    # Count tags to find top-N (matches preprocess.py exact logic)
    tag_counts = Counter()
    for tags in train_df['Tags'].fillna('').str.split():
        tag_counts.update(tags)
    
    # Take the top 'num_classes' tags to align with the prediction matrix columns
    top_tags = np.array([tag for tag, count in tag_counts.most_common(num_classes)], dtype=object)
    
    # Vocabulary size alignment check
    if len(top_tags) != num_classes:
        print(f"Warning: Vocabulary size mismatch. Reconstructed: {len(top_tags)}, Expected: {num_classes}")
        if len(top_tags) < num_classes:
            top_tags = np.append(top_tags, [""] * (num_classes - len(top_tags)))
    
    del train_df
    gc.collect()

    # Stage 3: Merge Model Predictions
    print("Stage 3/4: Averaging model probabilities and generating top-K tags...")
    # Simple arithmetic mean of probabilities across all participating models
    avg_test_probs = np.mean(list(all_test_preds.values()), axis=0)
    
    top_k_param = 3
    threshold = 0.2
    k = min(top_k_param, num_classes)
    
    # Vectorized extraction of top-K indices for efficiency
    # Step A: Get indices of top k elements (unsorted)
    top_indices = np.argpartition(-avg_test_probs, k-1, axis=1)[:, :k]
    row_indices = np.arange(n_test)[:, None]
    # Step B: Sort these selected k indices by actual probability values
    sorted_top_indices = top_indices[row_indices, np.argsort(-avg_test_probs[row_indices, top_indices], axis=1)]
    
    # Convert probability-thresholded indices into space-delimited tag strings
    model_test_preds = []
    for i in range(n_test):
        row_probs = avg_test_probs[i, sorted_top_indices[i]]
        row_tags = top_tags[sorted_top_indices[i]]
        # Keep only tags exceeding threshold to maintain precision
        selected = row_tags[row_probs > threshold]
        model_test_preds.append(" ".join(selected))

    # Stage 4: Apply Duplicate Override
    print("Stage 4/4: Applying duplicate lookup override logic...")
    # Load duplicate map prepared in load_data stage (maps (Title, Body) -> Tags)
    map_path = os.path.join(BASE_DATA_PATH, "leakage_prepared", "duplicate_map.pkl")
    with open(map_path, 'rb') as f:
        dup_map = pickle.load(f)

    # Load test features for lookup. We read 'nrows=n_test' to ensure alignment 
    # with the provided predictions, especially in validation mode.
    test_csv = os.path.join(BASE_DATA_PATH, "test.csv")
    test_info = pd.read_csv(test_csv, usecols=['Title', 'Body'], nrows=n_test, low_memory=True)
    test_info['Title'] = test_info['Title'].fillna('').astype('string')
    test_info['Body'] = test_info['Body'].fillna('').astype('string')
    
    final_preds = []
    override_count = 0
    
    # Iterate through features and model predictions simultaneously using zip for speed
    for title, body, m_pred in zip(test_info['Title'], test_info['Body'], model_test_preds):
        lookup_key = (title, body)
        if lookup_key in dup_map:
            # High-confidence override: match found in training data
            final_preds.append(str(dup_map[lookup_key]))
            override_count += 1
        else:
            # Fallback to model-generated prediction
            final_preds.append(m_pred)
            
    print(f"Ensemble completed. Overrode {override_count} samples with duplicate matches.")
    
    # Final sanity check on output length consistency
    if len(final_preds) != n_test:
        raise ValueError(f"Output length mismatch: generated {len(final_preds)} predictions for {n_test} test samples.")

    return final_preds