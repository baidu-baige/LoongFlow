import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any

BASE_DATA_PATH = "/root/workspace/evolux/output/mlebench/inaturalist-2019-fgvc6/prepared/public"
OUTPUT_DATA_PATH = "output/57a5d42d-daf8-4241-8197-424dd36c00a6/1/executor/output"

# Task-adaptive concrete type definitions
y = pd.Series     # Ground truth category_ids for the validation set
Predictions = np.ndarray    # Predictions (array of strings, each containing space-separated top-5 IDs)

def ensemble(
    all_val_preds: Dict[str, np.ndarray],
    all_test_preds: Dict[str, np.ndarray],
    y_val: y,
) -> Predictions:
    """
    Combines predictions from multiple models into a final submission-ready output.
    Uses Mean of Softmax Probabilities as the ensemble strategy and handles 
    category index mapping back to original competition IDs.

    Args:
        all_val_preds (Dict[str, np.ndarray]): Model names mapped to validation set logits.
        all_test_preds (Dict[str, np.ndarray]): Model names mapped to test set logits.
        y_val (y): Original category_ids for validation samples.

    Returns:
        Predictions: Final test set predictions as space-separated top-5 ID strings.
    """
    if not all_test_preds:
        raise ValueError("No model outputs provided for ensembling.")

    # Step 1: Combine model predictions using Mean of Softmax Probabilities
    model_names = list(all_test_preds.keys())
    num_test_samples, num_classes = all_test_preds[model_names[0]].shape
    
    # Initialize aggregated probabilities
    combined_probs = np.zeros((num_test_samples, num_classes), dtype=np.float32)
    
    for name in model_names:
        logits = all_test_preds[name]
        
        # Ensure numerical stability and handle any invalid values
        logits = np.nan_to_num(logits, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # Softmax computation
        shifted_logits = logits - np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(shifted_logits)
        model_probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        
        # Weighted mean (currently simple average as per combination guidance)
        combined_probs += model_probs
        
    combined_probs /= len(model_names)

    # Step 2: Reconstruct the category_id mapping (index -> competition category_id)
    # This logic must exactly match the upstream 'train_and_predict' mapping construction.
    def get_category_mapping():
        # Load JSON files to recover the exact order of samples used in the data loading stage
        with open(os.path.join(BASE_DATA_PATH, 'train2019.json'), 'r') as f:
            train_data = json.load(f)
        with open(os.path.join(BASE_DATA_PATH, 'val2019.json'), 'r') as f:
            val_data = json.load(f)
            
        # Replicate image-annotation alignment from load_data.py
        def extract_pool_ids(data):
            images_df = pd.DataFrame(data['images'])
            annotations_df = pd.DataFrame(data['annotations'])
            # Only images with annotations are included in training/validation pool
            df = images_df.merge(annotations_df, left_on='id', right_on='image_id')
            return df['category_id'].tolist()
            
        train_pool_ids = extract_pool_ids(train_data)
        val_pool_ids = extract_pool_ids(val_data)
        
        # The pool follows the concatenation order: train then validation
        full_ids_pool = train_pool_ids + val_pool_ids
        
        # Scenario 1: Full dataset (1010 species)
        full_unique_ids = sorted(list(set(full_ids_pool)))
        if len(full_unique_ids) == num_classes:
            return full_unique_ids
            
        # Scenario 2: validation_mode (subset of first 200 images)
        head_200_unique_ids = sorted(list(set(full_ids_pool[:200])))
        if len(head_200_unique_ids) == num_classes:
            return head_200_unique_ids
            
        # Propagate error if mapping cannot be accurately recovered
        raise RuntimeError(f"Dimensionality mismatch: Model predicts {num_classes} classes, "
                           f"but expected 1010 or {len(head_200_unique_ids)} classes.")

    mapping = get_mapping_from_data = get_category_mapping()

    # Step 3: Extract Top-5 predictions and format as strings
    # Sort indices by probability in descending order
    top5_indices = np.argsort(combined_probs, axis=1)[:, -5:][:, ::-1]
    
    final_test_outputs = []
    for row in top5_indices:
        # Convert dense indices to original category_id strings
        cat_labels = [str(mapping[idx]) for idx in row]
        # Join as space-separated string for submission format (id,predicted)
        final_test_outputs.append(" ".join(cat_labels))
        
    return np.array(final_test_outputs)