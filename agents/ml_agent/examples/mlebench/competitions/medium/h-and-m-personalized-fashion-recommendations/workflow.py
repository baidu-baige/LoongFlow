import os
import gc
import numpy as np
import cudf
import cupy as cp
from typing import Dict

# Import all component functions from provided files
from load_data import load_data
from preprocess import preprocess
from get_splitter import get_splitter
from train_and_predict import PREDICTION_ENGINES
from ensemble import ensemble

BASE_DATA_PATH = "/mnt/pfs/loongflow/devmachine/h20-01/evolux/output/mlebench/h-and-m-personalized-fashion-recommendations/prepared/public"
OUTPUT_DATA_PATH = "output/083259f0-3b2a-44c4-af6c-8557ef06ad6a/8/executor/output"
FUSION_SOURCE_PATH = "output/083259f0-3b2a-44c4-af6c-8557ef06ad6a/7/executor/output/submission.csv"

def workflow() -> dict:
    """
    Orchestrates the complete end-to-end machine learning pipeline in production mode.
    Integrates a fusion signal from a previous run as a feature for candidate ranking.
    """
    print("Starting production pipeline orchestration with Fusion Strategy...")

    # 1. Load full dataset
    # validation_mode=False loads the complete dataset for production training
    X_train_full, y_train_full, X_test_raw, test_ids = load_data(validation_mode=False)

    # 2. Load and Prepare Fusion Signal (ad7ccf68)
    fusion_df = None
    if os.path.exists(FUSION_SOURCE_PATH):
        print(f"Loading fusion signal from {FUSION_SOURCE_PATH}...")
        raw_fusion = cudf.read_csv(FUSION_SOURCE_PATH)
        # Convert prediction strings to long-format (customer_id, article_id)
        raw_fusion['article_id'] = raw_fusion['prediction'].str.split(' ')
        fusion_df = raw_fusion.explode('article_id')
        fusion_df['article_id'] = fusion_df['article_id'].astype('int32')
        # Assign dummy high score as a feature
        fusion_df['fusion_score_ad7ccf68'] = 1.0
        fusion_df = fusion_df[['customer_id', 'article_id', 'fusion_score_ad7ccf68']]
        print(f"Fusion signal loaded. Total pairs: {len(fusion_df)}")
        del raw_fusion
        gc.collect()
    else:
        print(f"Warning: Fusion source {FUSION_SOURCE_PATH} not found. Proceeding without fusion feature.")

    # 3. Set up data splitting strategy
    splitter = get_splitter(X_train_full, y_train_full)
    
    all_val_preds = {}
    all_test_preds = {}
    y_val_final = None

    # 4. Execute splits
    print(f"Executing splits using {splitter.__class__.__name__}...")
    for i, (train_idx, val_idx) in enumerate(splitter.split(X_train_full)):
        print(f"Processing split {i+1}...")
        
        # Isolate fold data
        X_train_fold = X_train_full.iloc[train_idx]
        y_train_fold = y_train_full.iloc[train_idx]
        X_val_fold = X_train_full.iloc[val_idx]
        y_val_fold = y_train_full.iloc[val_idx]
        
        # 5. Preprocess data for this fold
        X_tr_p, y_tr_p, X_va_p, y_va_p, X_te_p = preprocess(
            X_train_fold, 
            y_train_fold, 
            X_val_fold, 
            y_val_fold, 
            X_test_raw
        )
        
        # Memory Management after preprocessing
        del X_train_fold, y_train_fold, X_val_fold, y_val_fold
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()

        # 6. Integrate Fusion Feature
        if fusion_df is not None:
            print("Merging fusion signal into candidate features...")
            # Merge into test candidates
            X_te_p = X_te_p.merge(fusion_df, on=['customer_id', 'article_id'], how='left')
            X_te_p['fusion_score_ad7ccf68'] = X_te_p['fusion_score_ad7ccf68'].fillna(0.0).astype('float32')
            
            # Add feature to train/val sets (filled with 0 as signal is usually test-only)
            X_tr_p['fusion_score_ad7ccf68'] = cp.zeros(len(X_tr_p), dtype='float32')
            X_va_p['fusion_score_ad7ccf68'] = cp.zeros(len(X_va_p), dtype='float32')
        
        # Capture validation labels
        y_val_final = y_va_p
        
        # 7. Train and Predict
        for model_name, train_fn in PREDICTION_ENGINES.items():
            print(f"Fitting model: {model_name}")
            val_preds, test_preds = train_fn(X_tr_p, y_tr_p, X_va_p, y_va_p, X_te_p)
            
            all_val_preds[model_name] = val_preds
            all_test_preds[model_name] = test_preds
            
            # Memory management after each model execution
            gc.collect()
            cp.get_default_memory_pool().free_all_blocks()
            
        # Clear processed features for the fold
        del X_tr_p, y_tr_p, X_va_p, y_va_p, X_te_p
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()

    # 8. Ensemble predictions
    submission_df = ensemble(all_val_preds, all_test_preds, y_val_final)
    
    # 9. Compute prediction statistics
    def calculate_stats(preds_dict):
        if not preds_dict:
            return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
        
        scores = []
        for name in preds_dict:
            scores.append(cp.asarray(preds_dict[name]['score'].values))
        
        if len(scores) > 1:
            combined = cp.mean(cp.stack(scores), axis=0)
        else:
            combined = scores[0]
            
        return {
            "mean": float(cp.mean(combined)),
            "std": float(cp.std(combined)),
            "min": float(cp.min(combined)),
            "max": float(cp.max(combined)),
        }

    prediction_stats = {
        "oof": calculate_stats(all_val_preds),
        "test": calculate_stats(all_test_preds)
    }

    # 10. Prepare final output info
    submission_path = os.path.join(OUTPUT_DATA_PATH, "submission.csv")
    
    output_info = {
        "submission_file_path": submission_path,
        "prediction_stats": prediction_stats,
        "n_customers_predicted": int(len(submission_df))
    }

    print("Pipeline execution complete.")
    print(f"OOF Score Mean: {prediction_stats['oof']['mean']:.4f}")
    print(f"Test Score Mean: {prediction_stats['test']['mean']:.4f}")
    print(f"Submission saved to: {submission_path}")
    
    # Final cleanup
    del all_val_preds, all_test_preds, submission_df, X_train_full, y_train_full, X_test_raw, test_ids
    if fusion_df is not None: del fusion_df
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()
    
    return output_info