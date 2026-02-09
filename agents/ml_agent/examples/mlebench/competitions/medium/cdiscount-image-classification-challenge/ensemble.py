import os
import pandas as pd
import numpy as np
from typing import Dict, Any

BASE_DATA_PATH = "/mnt/pfs/loongflow/devmachine/h20-02/evolux/output/mlebench/cdiscount-image-classification-challenge/prepared/public"
OUTPUT_DATA_PATH = "output/96ece161-83e4-4d99-a688-ea7a2b1aa242/1/executor/output"

# Task-adaptive type definitions
y = pd.DataFrame
Predictions = np.ndarray

def ensemble(
    all_val_preds: Dict[str, Predictions],
    all_test_preds: Dict[str, Predictions],
    y_val: y,
) -> Predictions:
    """
    Combines image-level predictions from multiple models into product-level predictions
    using logit averaging.
    """
    print("Ensemble stage: Aggregating image-level logits to product-level predictions...")

    # 1. Load Mappings
    cat_csv = os.path.join(BASE_DATA_PATH, "category_names.csv")
    cat_df = pd.read_csv(cat_csv).sort_values('category_id').reset_index(drop=True)
    idx_to_cat_id = cat_df['category_id'].values

    # 2. Setup IDs for aggregation
    # We must recover the product IDs associated with the rows of the predictions.
    # Since load_data.py subsets using .iloc[:N] and resets index, we match by row count.
    train_idx_path = os.path.join(OUTPUT_DATA_PATH, "train_index_v2.parquet")
    test_idx_path = os.path.join(OUTPUT_DATA_PATH, "test_index_v2.parquet")
    
    train_index = pd.read_parquet(train_idx_path)
    test_index = pd.read_parquet(test_idx_path)

    # Note: all_val_preds corresponds to rows in y_val.
    # y_val.index contains the indices of the rows from the original train_index.
    val_ids = train_index.loc[y_val.index, '_id'].values
    
    # 3. Model Averaging (Logit Averaging)
    model_names = list(all_val_preds.keys())
    print(f"Ensembling models: {model_names}")
    
    avg_val_logits = np.mean([all_val_preds[name] for name in model_names], axis=0)
    avg_test_logits = np.mean([all_test_preds[name] for name in model_names], axis=0)

    # 4. Aggregation Function
    def aggregate_and_predict(logits, product_ids):
        """Aggregates image-level logits to product-level."""
        # Ensure lengths match strictly
        if len(logits) != len(product_ids):
            raise ValueError(f"Length mismatch: Logits ({len(logits)}) vs IDs ({len(product_ids)})")
            
        df = pd.DataFrame(logits)
        df['_id'] = product_ids
        
        # Mean logits across images of the same product
        prod_logits = df.groupby('_id', sort=False).mean()
        
        # Argmax and Category ID mapping
        prod_indices = prod_logits.values.argmax(axis=1)
        prod_cat_ids = idx_to_cat_id[prod_indices]
        
        # Return Series mapping product_id -> predicted category_id
        return pd.Series(prod_cat_ids, index=prod_logits.index)

    # 5. Validation Evaluation
    val_prod_preds = aggregate_and_predict(avg_val_logits, val_ids)
    
    # Ground truth mapping
    val_metadata = pd.DataFrame({'_id': val_ids, 'cat_idx': y_val['cat_idx'].values})
    val_prod_gt_idx = val_metadata.groupby('_id', sort=False)['cat_idx'].first()
    val_prod_gt_ids = idx_to_cat_id[val_prod_gt_idx.values]
    
    accuracy = (val_prod_preds.values == val_prod_gt_ids).mean()
    print(f"Product-level Validation Accuracy: {accuracy:.4f}")

    # 6. Test Prediction
    # Recover test IDs for the samples predicted
    num_test_samples = len(avg_test_logits)
    test_ids_subset = test_index['_id'].values[:num_test_samples]
    
    test_prod_preds = aggregate_and_predict(avg_test_logits, test_ids_subset)
    
    # Map product-level decisions back to image-level rows to satisfy return length requirement
    test_results_map = test_prod_preds.to_dict()
    final_test_preds = np.array([test_results_map[pid] for pid in test_ids_subset])

    print(f"Ensemble complete. Final test predictions shape: {final_test_preds.shape}")
    return final_test_preds