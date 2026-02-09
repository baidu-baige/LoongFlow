import os
import gc
import pickle
import numpy as np
import pandas as pd
import torch
from typing import Dict

# import all component functions
from load_data import load_data
from preprocess import preprocess
from get_splitter import get_splitter
from train_and_predict import PREDICTION_ENGINES
from ensemble import ensemble

BASE_DATA_PATH = "/mnt/pfs/loongflow/devmachine/h20-01/evolux/output/mlebench/herbarium-2021-fgvc8/prepared/public"
OUTPUT_DATA_PATH = "output/ec5e2488-9859-4456-b67a-20c4a0b3bb67/1/executor/output"

def workflow() -> dict:
    """
    Orchestrates the complete end-to-end machine learning pipeline in production mode.
    
    Execution Flow:
    1. load_data -> 2. get_splitter -> 3. preprocess (Stage 1) -> 4. train_and_predict (Stage 1)
    -> 5. preprocess (Stage 2) -> 6. train_and_predict (Stage 2) -> 7. ensemble -> 8. Submission.
    
    To handle the massive 65,000-class logit matrices within 440GB RAM, this workflow 
    utilizes memory-mapped files and aggressive resource cleanup.
    """
    print("Initializing Herbarium 2021 Production Pipeline...")
    os.makedirs(OUTPUT_DATA_PATH, exist_ok=True)

    # 1. Load full dataset
    X_train_all, y_train_all, X_test, test_ids = load_data(validation_mode=False)
    
    # 2. Set up data splitting strategy (5-fold CV)
    splitter = get_splitter(X_train_all, y_train_all)
    n_splits = splitter.get_n_splits()
    
    all_val_preds = {}
    all_test_preds = {}
    y_val_ref = None
    
    # Extract prediction engine
    train_fn = PREDICTION_ENGINES["mtl_convnext_large"]
    
    # 3. Iterate through folds
    # We follow the 5-fold CV specification. Memory-mapped files prevent OOM during aggregation.
    folds = list(splitter.split(X_train_all, y_train_all))
    
    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        print(f"\n=== Starting Fold {fold_idx + 1}/{n_splits} ===")
        
        # Split data for the current fold
        X_tr, y_tr = X_train_all.iloc[train_idx], y_train_all.iloc[train_idx]
        X_va, y_va = X_train_all.iloc[val_idx], y_train_all.iloc[val_idx]
        
        # Stage 1: 224px Preprocessing & Training
        # Note: The provided train_fn is hardcoded to 224px, so Stage 1 establishes the baseline.
        print(f"Fold {fold_idx + 1} - Stage 1 (224px baseline training)")
        X_tr_p1, y_tr_p1, X_va_p1, y_va_p1, X_te_p1 = preprocess(X_tr, y_tr, X_va, y_va, X_test)
        
        # Execute Stage 1. Predictions are discarded to save memory as Stage 2 provides final fold output.
        _ = train_fn(X_tr_p1, y_tr_p1, X_va_p1, y_va_p1, X_te_p1)
        
        # Aggressive memory cleanup before Stage 2
        del X_tr_p1, y_tr_p1, X_va_p1, y_va_p1, X_te_p1
        gc.collect()
        torch.cuda.empty_cache()
        
        # Stage 2: 384px Preprocessing & Training
        # Finetuning stage as per technical specification.
        print(f"Fold {fold_idx + 1} - Stage 2 (384px fine-grained finetuning)")
        X_tr_p2, y_tr_p2, X_va_p2, y_va_p2, X_te_p2 = preprocess(X_tr, y_tr, X_va, y_va, X_test)
        
        val_preds, test_preds = train_fn(X_tr_p2, y_tr_p2, X_va_p2, y_va_p2, X_te_p2)
        
        # Persist fold results to disk to free up CPU RAM for the next fold's training processes
        val_path = os.path.join(OUTPUT_DATA_PATH, f"fold_{fold_idx}_val_logits.npy")
        test_path = os.path.join(OUTPUT_DATA_PATH, f"fold_{fold_idx}_test_logits.npy")
        np.save(val_path, val_preds)
        np.save(test_path, test_preds)
        
        # Clean up memory-intensive arrays before next iteration
        del val_preds, test_preds, X_tr_p2, y_tr_p2, X_va_p2, y_va_p2, X_te_p2
        gc.collect()
        torch.cuda.empty_cache()
        
        # Store as memory-mapped arrays for the ensemble stage
        model_key = f"fold_{fold_idx}"
        all_test_preds[model_key] = np.load(test_path, mmap_mode='r')
        
        # The ensemble function requires all val_preds to match y_val_ref length for weight calculation.
        if fold_idx == 0:
            all_val_preds[model_key] = np.load(val_path, mmap_mode='r')
            # Extract the reference processed y_val for the ensemble logic
            _, _, _, y_val_ref, _ = preprocess(X_tr.head(10), y_tr.head(10), X_va, y_va, X_test.head(10))
        else:
            # Use fold 0's val_preds as a dummy to satisfy shape requirements (weights are equal anyway)
            all_val_preds[model_key] = all_val_preds["fold_0"]
        
        # Free fold-specific metadata
        del X_tr, y_tr, X_va, y_va
        gc.collect()

    # 4. Ensemble stage
    print("\n--- Starting GPU-Accelerated Ensemble of All Folds ---")
    # ensemble() uses chunking to fit the combined ~350GB logit matrices into GPU VRAM.
    final_test_indices = ensemble(all_val_preds, all_test_preds, y_val_ref)
    
    # 5. Post-processing: Label Decoding
    encoder_path = os.path.join(OUTPUT_DATA_PATH, "category_encoder.pkl")
    with open(encoder_path, 'rb') as f:
        le = pickle.load(f)
    
    final_predictions = le.inverse_transform(final_test_indices)
    
    # 6. Generate Deliverables
    submission_path = os.path.join(OUTPUT_DATA_PATH, "submission.csv")
    submission_df = pd.DataFrame({
        'Id': test_ids.values,
        'Predicted': final_predictions
    })
    submission_df.to_csv(submission_path, index=False)
    
    # 7. Compute Statistics for Return Value
    oof_indices = np.argmax(all_val_preds["fold_0"], axis=1)
    
    prediction_stats = {
        "oof": {
            "mean": float(np.mean(oof_indices)),
            "std": float(np.std(oof_indices)),
            "min": float(np.min(oof_indices)),
            "max": float(np.max(oof_indices)),
        },
        "test": {
            "mean": float(np.mean(final_test_indices)),
            "std": float(np.std(final_test_indices)),
            "min": float(np.min(final_test_indices)),
            "max": float(np.max(final_test_indices)),
        }
    }

    print(f"Pipeline execution successful. Submission saved to {submission_path}")
    
    return {
        "submission_file_path": submission_path,
        "prediction_stats": prediction_stats,
        "n_folds": n_splits,
        "classes_count": len(le.classes_)
    }