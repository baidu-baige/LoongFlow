import os
import pandas as pd
import numpy as np
import torch
from load_data import load_data
from preprocess import preprocess
from get_splitter import get_splitter
from train_and_predict import PREDICTION_ENGINES
from ensemble import ensemble

BASE_DATA_PATH = "/mnt/pfs/loongflow/devmachine/h20-04/evolux/output/mlebench/hotel-id-2021-fgvc8/prepared/public"
OUTPUT_DATA_PATH = "output/6c275358-248b-46e3-a3f8-feb17fef7b7f/3/executor/output"

def workflow() -> dict:
    """
    Orchestrates the complete end-to-end machine learning pipeline in production mode.
    Integrates loading, splitting, preprocessing, training, and ensembling stages.
    """
    # 1. Load the full dataset (production mode)
    print("Stage 1: Loading complete dataset for production...")
    X_train_full, y_train_full, X_test_full, test_ids = load_data(validation_mode=False)

    # 2. Define the splitting strategy
    print("Stage 2: Initializing robust validation strategy...")
    splitter = get_splitter(X_train_full, y_train_full)

    # 3. Training and Inference Loop
    # Following the Technical Specification, we execute a single fold strategy 
    # to ensure completion within competition time limits while maximizing gallery size.
    print("Stage 3: Starting fold-based training and inference...")
    
    # Extract indices for the first fold
    train_idx, val_idx = next(splitter.split(X_train_full, y_train_full))
    
    X_tr, y_tr = X_train_full.iloc[train_idx], y_train_full.iloc[train_idx]
    X_va, y_va = X_train_full.iloc[val_idx], y_train_full.iloc[val_idx]
    
    # 3a. Preprocessing for the current fold
    # This defines the 384x384 augmentation pipelines and saves them to disk.
    print(f"Executing preprocessing for fold (Train: {len(X_tr)}, Val: {len(X_va)})...")
    X_tr_proc, y_tr_proc, X_va_proc, y_va_proc, X_te_proc = preprocess(
        X_tr, y_tr, X_va, y_va, X_test_full
    )
    
    # 3b. Train and Predict using the retrieval engine
    # We use the registered retrieval_arcface_efficientnet engine.
    model_key = "retrieval_arcface_efficientnet"
    if model_key not in PREDICTION_ENGINES:
        raise KeyError(f"Registry error: '{model_key}' not found in PREDICTION_ENGINES.")
    
    train_fn = PREDICTION_ENGINES[model_key]
    print(f"Training and generating predictions using: {model_key}...")
    val_preds, test_preds = train_fn(X_tr_proc, y_tr_proc, X_va_proc, y_va_proc, X_te_proc)
    
    # Store predictions for the ensemble stage
    all_val_preds = {model_key: val_preds}
    all_test_preds = {model_key: test_preds}

    # 4. Ensembling and Ranking (Global Similarity Search)
    # This stage re-extracts test embeddings and performs GPU-accelerated k-NN search 
    # against the full training gallery to generate the final space-delimited hotel IDs.
    print("Stage 4: Consolidating predictions via ensembling and GPU k-NN search...")
    final_ranking_series = ensemble(all_val_preds, all_test_preds, y_va_proc)

    # 5. Generate Submission Artifact
    print("Stage 5: Formatting and saving final submission.csv...")
    # Alignment check: Ensure the number of predictions matches the number of test images.
    if len(final_ranking_series) != len(test_ids):
        # The ensemble function might have subsetted test_df if validation_mode was used upstream.
        # However, for production (validation_mode=False), they must match.
        print(f"Warning: Alignment mismatch. Results: {len(final_ranking_series)}, Expected: {len(test_ids)}")

    submission_df = pd.DataFrame({
        'image': test_ids.values[:len(final_ranking_series)],
        'hotel_id': final_ranking_series.values
    })
    
    os.makedirs(OUTPUT_DATA_PATH, exist_ok=True)
    submission_file_path = os.path.join(OUTPUT_DATA_PATH, 'submission.csv')
    submission_df.to_csv(submission_file_path, index=False)

    # 6. Compute Prediction Statistics
    # Statistics are computed on the raw top-k prediction index arrays.
    print("Stage 6: Computing final task deliverables and statistics...")
    prediction_stats = {
        "oof": {
            "mean": float(np.mean(val_preds)),
            "std": float(np.std(val_preds)),
            "min": float(np.min(val_preds)),
            "max": float(np.max(val_preds)),
        },
        "test": {
            "mean": float(np.mean(test_preds)),
            "std": float(np.std(test_preds)),
            "min": float(np.min(test_preds)),
            "max": float(np.max(test_preds)),
        }
    }

    # Verify submission file existence before returning
    if not os.path.exists(submission_file_path):
        raise RuntimeError(f"Pipeline failed to produce submission file at {submission_file_path}")

    print(f"Workflow complete. Final submission saved to {submission_file_path}")
    
    return {
        "submission_file_path": submission_file_path,
        "prediction_stats": prediction_stats
    }