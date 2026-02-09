import os
import gc
import numpy as np
import pandas as pd
from typing import Dict

# Import component functions as specified
from load_data import load_data
from preprocess import preprocess
from get_splitter import get_splitter
from train_and_predict import PREDICTION_ENGINES
from ensemble import ensemble

BASE_DATA_PATH = "/mnt/pfs/loongflow/devmachine/h20-03/evolux/output/mlebench/facebook-recruiting-iii-keyword-extraction/prepared/public"
OUTPUT_DATA_PATH = "output/90689814-166b-4e4f-a971-355572d18239/1/executor/output"

def workflow() -> dict:
    """
    Orchestrates the complete end-to-end machine learning pipeline in production mode.
    Handles data loading, splitting, preprocessing, model training, and ensembling
    while optimizing for memory usage on a large-scale dataset (5.4M rows).
    """
    print("Starting production workflow...")
    
    # Ensure output directory exists
    os.makedirs(OUTPUT_DATA_PATH, exist_ok=True)
    
    # 1. Load full dataset
    # Returns: X_train (DF), y_train (Series), X_test (DF), test_ids (Series)
    X_train_full, y_train_full, X_test_df, test_ids = load_data(validation_mode=False)
    
    # 2. Set up data splitting strategy
    # Configured for a single 90/10 split (ShuffleSplit)
    splitter = get_splitter(X_train_full, y_train_full)
    
    # Storage for cross-model predictions
    all_val_preds: Dict[str, np.ndarray] = {}
    all_test_preds: Dict[str, np.ndarray] = {}
    
    # 3. Process the split (Expects exactly 1 fold as per get_splitter specification)
    print("Beginning data splitting and preprocessing...")
    for train_idx, val_idx in splitter.split(X_train_full, y_train_full):
        # Extract slices
        X_train_fold = X_train_full.iloc[train_idx]
        y_train_fold = y_train_full.iloc[train_idx]
        X_val_fold = X_train_full.iloc[val_idx]
        y_val_fold = y_train_full.iloc[val_idx]
        
        # Preprocess features and labels into sparse matrices
        # Returns: X_train_p, y_train_p, X_val_p, y_val_p, X_test_p
        X_train_p, y_train_p, X_val_p, y_val_p, X_test_p = preprocess(
            X_train_fold, y_train_fold, X_val_fold, y_val_fold, X_test_df
        )
        
        # Immediate memory cleanup of massive raw text DataFrames
        del X_train_full, y_train_full, X_test_df, X_train_fold, y_train_fold, X_val_fold, y_val_fold
        gc.collect()
        
        # 4. Train each registered model engine
        for engine_name, train_fn in PREDICTION_ENGINES.items():
            print(f"Training prediction engine: {engine_name}...")
            # Each engine returns dense probability matrices (float32)
            v_preds, t_preds = train_fn(X_train_p, y_train_p, X_val_p, y_val_p, X_test_p)
            all_val_preds[engine_name] = v_preds
            all_test_preds[engine_name] = t_preds
            
        # Archive y_val_p for the ensemble stage (it's a sparse indicator matrix)
        saved_y_val_p = y_val_p
        
        # Cleanup processed training data to release RAM for ensembling
        del X_train_p, y_train_p, X_val_p, X_test_p
        gc.collect()
        
        # Break after first fold as per ShuffleSplit(n_splits=1)
        break

    # 5. Ensemble predictions
    # This combines model probabilities and applies the duplicate lookup override
    print("Executing ensemble and duplicate override logic...")
    final_tags = ensemble(all_val_preds, all_test_preds, saved_y_val_p)
    
    # 6. Compute prediction statistics
    # Stats are calculated on the averaged probability estimates across all models
    print("Calculating prediction statistics...")
    combined_val_probs = np.mean(list(all_val_preds.values()), axis=0) if all_val_preds else np.array([0])
    combined_test_probs = np.mean(list(all_test_preds.values()), axis=0) if all_test_preds else np.array([0])
    
    prediction_stats = {
        "oof": {
            "mean": float(np.mean(combined_val_probs)),
            "std": float(np.std(combined_val_probs)),
            "min": float(np.min(combined_val_probs)),
            "max": float(np.max(combined_val_probs)),
        },
        "test": {
            "mean": float(np.mean(combined_test_probs)),
            "std": float(np.std(combined_test_probs)),
            "min": float(np.min(combined_test_probs)),
            "max": float(np.max(combined_test_probs)),
        }
    }
    
    # 7. Generate deliverables
    submission_path = os.path.join(OUTPUT_DATA_PATH, "submission.csv")
    print(f"Saving final submission to {submission_path}...")
    
    submission_df = pd.DataFrame({
        "Id": test_ids,
        "Tags": final_tags
    })
    submission_df.to_csv(submission_path, index=False)
    
    print("Workflow complete.")
    
    return {
        "submission_file_path": submission_path,
        "prediction_stats": prediction_stats
    }