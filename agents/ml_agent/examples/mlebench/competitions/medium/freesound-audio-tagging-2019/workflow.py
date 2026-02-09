import os
import numpy as np
import pandas as pd
from load_data import load_data
from preprocess import preprocess
from get_splitter import get_splitter
from train_and_predict import PREDICTION_ENGINES
from ensemble import ensemble

BASE_DATA_PATH = "/mnt/pfs/loongflow/devmachine/h20-04/evolux/output/mlebench/freesound-audio-tagging-2019/prepared/public"
OUTPUT_DATA_PATH = "output/41d6445d-6924-4bdd-a657-0e220176048a/2/executor/output"

def workflow() -> dict:
    """
    Orchestrates the complete end-to-end machine learning pipeline in production mode.
    
    This workflow executes:
    1. Full data loading (curated and noisy subsets).
    2. Multi-label stratified 5-fold cross-validation.
    3. Per-fold audio preprocessing (Z-score normalization and Log-Mel transformation).
    4. 2-stage training (Noisy+Curated -> Fine-tune Curated) using EfficientNet-B2 on dual GPUs.
    5. Ensemble of fold predictions and generation of final submission.
    """
    print("Starting production workflow for Freesound Audio Tagging 2019...")

    # 1. Load full dataset
    # Returns metadata DataFrames and the multi-hot target matrix
    X_train_all, y_train_all, X_test_all, test_ids = load_data(validation_mode=False)
    label_cols = y_train_all.columns.tolist()
    
    # 2. Set up data splitting strategy
    # Stratifies on curated data while including noisy data in every training fold
    splitter = get_splitter(X_train_all, y_train_all)
    n_splits = splitter.get_n_splits()
    
    all_test_preds = {}
    # Container for out-of-fold predictions to evaluate model consistency
    oof_preds = np.zeros_like(y_train_all.values, dtype=np.float32)
    
    # 3. Execution of Cross-Validation Loop
    for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(X_train_all, y_train_all)):
        print(f"\n--- Executing Fold {fold_idx + 1}/{n_splits} ---")
        
        # Partition metadata for the current fold
        X_tr, y_tr = X_train_all.iloc[train_idx], y_train_all.iloc[train_idx]
        X_va, y_va = X_train_all.iloc[val_idx], y_train_all.iloc[val_idx]
        
        # a. Preprocess audio: Transform wav to Log-Mel Spectrograms
        # Normalization is fitted strictly on the training subset of the current fold
        X_tr_p, y_tr_p, X_va_p, y_va_p, X_te_p = preprocess(
            X_tr, y_tr, X_va, y_va, X_test_all
        )
        
        # b. Train and Predict using the EfficientNet-B2 engine
        # Logic: Stage 1 (All data) -> Stage 2 (Curated data only)
        # Hardware: DistributedDataParallel on 2 NVIDIA H20 GPUs
        val_p, test_p = PREDICTION_ENGINES["efficientnet_b2_two_stage"](
            X_tr_p, y_tr_p, X_va_p, y_va_p, X_te_p
        )
        
        # c. Record results for ensembling and OOF analysis
        oof_preds[val_idx] = val_p
        all_test_preds[f"fold_{fold_idx}"] = test_p
        
        print(f"Fold {fold_idx + 1} processing complete.")

    # 4. Ensemble stage: Combine fold predictions via arithmetic averaging
    # The ensemble function handles final probability clipping and validity checks
    final_test_preds = ensemble(
        all_val_preds={"global_oof": oof_preds},
        all_test_preds=all_test_preds,
        y_val=y_train_all.values
    )
    
    # 5. Generate and save the final submission file
    os.makedirs(OUTPUT_DATA_PATH, exist_ok=True)
    submission_df = pd.DataFrame(final_test_preds, columns=label_cols)
    submission_df.insert(0, 'fname', test_ids.values)
    
    submission_file_path = os.path.join(OUTPUT_DATA_PATH, "submission.csv")
    submission_df.to_csv(submission_file_path, index=False)
    print(f"Final submission saved to: {submission_file_path}")
    
    # 6. Compute prediction statistics for the final report
    # We filter OOF stats to curated samples only, as noisy samples were not part of validation
    curated_mask = X_train_all['is_curated'].values
    valid_oof = oof_preds[curated_mask]
    
    prediction_stats = {
        "oof": {
            "mean": float(np.mean(valid_oof)),
            "std": float(np.std(valid_oof)),
            "min": float(np.min(valid_oof)),
            "max": float(np.max(valid_oof)),
        },
        "test": {
            "mean": float(np.mean(final_test_preds)),
            "std": float(np.std(final_test_preds)),
            "min": float(np.min(final_test_preds)),
            "max": float(np.max(final_test_preds)),
        }
    }
    
    # 7. Package deliverables
    output_info = {
        "submission_file_path": submission_file_path,
        "prediction_stats": prediction_stats,
        "metadata": {
            "n_folds": n_splits,
            "model_architecture": "EfficientNet-B2",
            "training_strategy": "2-Stage Fine-tuning"
        }
    }
    
    print("Production workflow finished successfully.")
    return output_info