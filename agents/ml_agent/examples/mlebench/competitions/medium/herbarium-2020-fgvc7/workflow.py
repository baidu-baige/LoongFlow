import os
import torch
import pandas as pd
import numpy as np
import joblib
from typing import Dict, List, Any

# Import all component functions
from load_data import load_data
from preprocess import preprocess
from get_splitter import get_splitter
from train_and_predict import PREDICTION_ENGINES
from ensemble import ensemble

# Pipeline configuration constants
BASE_DATA_PATH = "/mnt/pfs/loongflow/devmachine/h20-02/evolux/output/mlebench/herbarium-2020-fgvc7/prepared/public"
OUTPUT_DATA_PATH = "output/cf84c6d3-8647-45ca-b9ff-02e7ed67cf5b/1/executor/output"

def workflow() -> dict:
    """
    Orchestrates the complete end-to-end machine learning pipeline in production mode.
    """
    print("Starting production workflow for Herbarium 2020 FGVC7...")
    os.makedirs(OUTPUT_DATA_PATH, exist_ok=True)

    # 1. Load full dataset
    print("Loading full dataset...")
    X_train_raw, y_train_raw, X_test_raw, test_ids = load_data(validation_mode=False)

    # 2. Set up data splitting strategy
    print("Initializing splitter...")
    splitter = get_splitter(X_train_raw, y_train_raw)

    all_test_preds = {}
    all_oof_preds = []
    
    # We will use this to store the mapping for OOF stat calculation later
    mapping_data = None

    # 3. Cross-Validation Loop
    fold = 0
    for train_idx, val_idx in splitter.split(X_train_raw, y_train_raw):
        print(f"--- Processing Fold {fold + 1} ---")
        
        # Split data
        X_tr, X_val = X_train_raw.iloc[train_idx].copy(), X_train_raw.iloc[val_idx].copy()
        y_tr, y_val = y_train_raw.iloc[train_idx].copy(), y_train_raw.iloc[val_idx].copy()

        # Apply preprocessing
        print(f"Preprocessing Fold {fold + 1}...")
        X_tr_p, y_tr_p, X_val_p, y_val_p, X_test_p = preprocess(
            X_tr, y_tr, X_val, y_val, X_test_raw
        )
        
        # Load mapping (consistent across folds due to StratifiedSplit)
        if mapping_data is None:
            mapping_path = os.path.join(OUTPUT_DATA_PATH, "category_mapping.joblib")
            mapping_data = joblib.load(mapping_path)

        # Train each model defined in the registry
        for engine_name, train_fn in PREDICTION_ENGINES.items():
            print(f"Training model: {engine_name} on fold {fold + 1}...")
            # train_fn returns (val_preds, test_preds) as np.ndarray
            val_preds, test_preds = train_fn(X_tr_p, y_tr_p, X_val_p, y_val_p, X_test_p)
            
            # Save test predictions for ensembling
            all_test_preds[f"{engine_name}_fold_{fold}"] = test_preds
            
            # Map OOF predictions back to original category_id for statistics
            idx_to_cat = mapping_data['idx_to_cat']
            max_idx = max(idx_to_cat.keys())
            lookup_table = np.zeros(max_idx + 1, dtype=np.int64)
            for idx, cat_id in idx_to_cat.items():
                lookup_table[idx] = cat_id
            
            # Process OOF predictions (hard labels)
            v_idx = np.clip(val_preds, 0, max_idx)
            oof_cat_ids = lookup_table[v_idx]
            all_oof_preds.append(oof_cat_ids)

        fold += 1

    # 4. Ensemble predictions from all folds and models
    print("Executing ensemble for final test predictions...")
    # The ensemble function uses category_mapping.joblib internally to map to category_id
    # We provide a dummy y_val from the last fold batch just for the signature (not used for test aggregation)
    final_test_predictions = ensemble(
        all_val_preds={}, # Not strictly needed for final test output generation
        all_test_preds=all_test_preds,
        y_val=y_val_p 
    )

    # 5. Compute prediction statistics
    print("Computing prediction statistics...")
    combined_oof = np.concatenate(all_oof_preds)
    
    prediction_stats = {
        "oof": {
            "mean": float(np.mean(combined_oof)),
            "std": float(np.std(combined_oof)),
            "min": float(np.min(combined_oof)),
            "max": float(np.max(combined_oof)),
        },
        "test": {
            "mean": float(np.mean(final_test_predictions)),
            "std": float(np.std(final_test_predictions)),
            "min": float(np.min(final_test_predictions)),
            "max": float(np.max(final_test_predictions)),
        }
    }

    # 6. Generate submission file
    print("Generating submission.csv...")
    submission_df = pd.DataFrame({
        'Id': test_ids.values,
        'Predicted': final_test_predictions
    })
    
    submission_file_path = os.path.join(OUTPUT_DATA_PATH, "submission.csv")
    submission_df.to_csv(submission_file_path, index=False)

    # 7. Finalize output info
    output_info = {
        "submission_file_path": submission_file_path,
        "prediction_stats": prediction_stats,
    }
    
    print(f"Workflow complete. Submission saved to {submission_file_path}")
    return output_info