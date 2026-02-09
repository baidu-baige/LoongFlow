import os
import pandas as pd
import numpy as np
import torch
import gc
from load_data import load_data
from preprocess import preprocess
from get_splitter import get_splitter
from train_and_predict import PREDICTION_ENGINES
from ensemble import ensemble

BASE_DATA_PATH = "/mnt/pfs/loongflow/devmachine/h20-02/evolux/output/mlebench/cdiscount-image-classification-challenge/prepared/public"
OUTPUT_DATA_PATH = "output/96ece161-83e4-4d99-a688-ea7a2b1aa242/1/executor/output"

def workflow() -> dict:
    """
    Orchestrates the complete end-to-end machine learning pipeline in production mode.
    """
    print("Starting production workflow...")
    os.makedirs(OUTPUT_DATA_PATH, exist_ok=True)

    # 1. Load full dataset
    # This generates/loads indexing for 52GB BSON and 15M+ images.
    X_train_full, y_train_full, X_test, test_ids = load_data(validation_mode=False)
    
    # 2. Set up data splitting strategy
    # Product-Stratified Shuffle Split ensures no leakage and maintains class distribution.
    splitter = get_splitter(X_train_full, y_train_full)
    
    # Storage for predictions across folds (though n_splits=1 based on get_splitter)
    all_val_logits = {}
    all_test_logits = {}
    y_val_combined = []
    
    # 3. Execution loop (Processing single fold as per Technical Spec)
    for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(X_train_full, y_train_full)):
        print(f"Processing Fold {fold_idx + 1}...")
        
        # Split data for this fold
        X_train_fold = X_train_full.iloc[train_idx]
        y_train_fold = y_train_full.iloc[train_idx]
        X_val_fold = X_train_full.iloc[val_idx]
        y_val_fold = y_train_full.iloc[val_idx]
        
        # Clean up global references to save RAM for GPU training
        # These are no longer needed after splitting
        if fold_idx == splitter.get_n_splits() - 1:
            del X_train_full
            del y_train_full
            gc.collect()

        # Preprocess metadata for efficient BSON loading
        X_train_proc, y_train_proc, X_val_proc, y_val_proc, X_test_proc = preprocess(
            X_train_fold, y_train_fold, X_val_fold, y_val_fold, X_test
        )
        
        # Train and Predict (Multi-Task EfficientNet-B2)
        # Uses DDP and mixed precision as per baseline requirements
        model_name = "multitask_efficientnet_b2"
        train_fn = PREDICTION_ENGINES[model_name]
        
        val_logits, test_logits = train_fn(
            X_train_proc, y_train_proc, X_val_proc, y_val_proc, X_test_proc
        )
        
        # Aggregate results
        all_val_logits[model_name] = val_logits
        all_test_logits[model_name] = test_logits
        y_val_combined.append(y_val_proc)
        
        # Break after first fold if only one split is requested
        if fold_idx + 1 >= splitter.get_n_splits():
            break

    # Consolidate validation targets for ensemble
    y_val_final = pd.concat(y_val_combined)
    
    # 4. Ensemble Predictions
    # Aggregates image-level logits to product-level category IDs
    final_test_preds = ensemble(all_val_logits, all_test_logits, y_val_final)
    
    # 5. Generate Submission File
    # Create product-level submission from image-level predictions
    # final_test_preds matches the row order of X_test_proc
    submission_path = os.path.join(OUTPUT_DATA_PATH, "submission.csv")
    print(f"Creating submission file at {submission_path}...")
    
    submission_df = pd.DataFrame({
        '_id': X_test_proc['_id'].values,
        'category_id': final_test_preds
    }).drop_duplicates(subset=['_id'])
    
    submission_df.to_csv(submission_path, index=False)
    
    # 6. Compute Prediction Statistics
    # For 'oof', we calculate the argmax of the aggregated val logits
    # Since we have one model, we can just use all_val_logits[model_name]
    val_id_preds = np.argmax(all_val_logits["multitask_efficientnet_b2"], axis=1)
    
    prediction_stats = {
        "oof": {
            "mean": float(np.mean(val_id_preds)),
            "std": float(np.std(val_id_preds)),
            "min": float(np.min(val_id_preds)),
            "max": float(np.max(val_id_preds)),
        },
        "test": {
            "mean": float(np.mean(final_test_preds)),
            "std": float(np.std(final_test_preds)),
            "min": float(np.min(final_test_preds)),
            "max": float(np.max(final_test_preds)),
        }
    }
    
    # 7. Final Deliverables
    output_info = {
        "submission_file_path": submission_path,
        "prediction_stats": prediction_stats,
        "model_architecture": "MultiTask EfficientNet-B2",
        "num_test_products": len(submission_df)
    }
    
    print("Workflow complete.")
    return output_info

if __name__ == "__main__":
    workflow()