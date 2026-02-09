import os
import pandas as pd
import numpy as np
import torch
from load_data import load_data
from preprocess import preprocess
from get_splitter import get_splitter
from train_and_predict import PREDICTION_ENGINES
from ensemble import ensemble

# Path configuration
BASE_DATA_PATH = "/mnt/pfs/loongflow/devmachine/h20-02/evolux/output/mlebench/multi-modal-gesture-recognition/prepared/public"
OUTPUT_DATA_PATH = "output/7d9b4fa5-39b3-4a58-b088-17228f41073d/11/executor/output"

def workflow() -> dict:
    """
    Orchestrates the complete end-to-end machine learning pipeline in production mode.
    """
    # 0. Setup Environment
    os.makedirs(OUTPUT_DATA_PATH, exist_ok=True)
    print(f"Starting production workflow. Outputs will be saved to: {OUTPUT_DATA_PATH}")

    # 1. Load full dataset (Production Mode)
    # MUST call load_data(validation_mode=False) for the complete dataset
    print("Step 1: Loading complete dataset...")
    X_train_all, y_train_all, X_test_all, test_ids_all = load_data(validation_mode=False)

    # 2. Set up data splitting strategy
    print("Step 2: Initializing ShuffledGroupKFold splitter...")
    splitter = get_splitter(X_train_all, y_train_all)
    n_splits = splitter.get_n_splits(X_train_all, y_train_all)
    
    # Store Out-of-Fold (OOF) probabilities and per-fold test probabilities
    all_oof_probs = [None] * len(X_train_all)
    fold_test_probs_list = []

    # 3. Model Engine Selection
    model_engine_key = "mstc_gru_fusion"
    if model_engine_key not in PREDICTION_ENGINES:
        raise KeyError(f"Selected model engine '{model_engine_key}' not found in PREDICTION_ENGINES registry.")
    
    train_fn = PREDICTION_ENGINES[model_engine_key]
    
    # 4. K-Fold Cross-Validation Loop
    print(f"Step 3: Starting {n_splits}-fold cross-validation using {model_engine_key}...")

    for fold, (train_idx, val_idx) in enumerate(splitter.split(X_train_all, y_train_all)):
        print(f"\n--- Fold {fold + 1}/{n_splits} Execution ---")
        
        # Partition raw data for current fold
        X_train_fold = [X_train_all[i] for i in train_idx]
        y_train_fold = [y_train_all[i] for i in train_idx]
        X_val_fold = [X_train_all[i] for i in val_idx]
        y_val_fold = [y_train_all[i] for i in val_idx]
        
        # a. Preprocessing: Feature extraction, normalization, and alignment
        print(f"Preprocessing data for fold {fold + 1}...")
        X_tr_proc, y_tr_proc, X_val_proc, y_val_proc, X_te_proc = preprocess(
            X_train_fold, y_train_fold, X_val_fold, y_val_fold, X_test_all
        )
        
        # b. Training and Inference: Run on GPUs
        print(f"Training and generating probabilities for fold {fold + 1}...")
        val_probs, test_probs = train_fn(
            X_tr_proc, y_tr_proc, X_val_proc, y_val_proc, X_te_proc
        )
        
        # c. Record results
        for i, original_idx in enumerate(val_idx):
            all_oof_probs[original_idx] = val_probs[i]
        
        fold_test_probs_list.append(test_probs)
        
        # Resource cleanup
        del X_tr_proc, X_val_proc, X_te_proc
        torch.cuda.empty_cache()

    # 5. Ensemble Phase
    print("\nStep 4: Combining predictions via logit-averaging and temporal ensembling...")
    all_val_preds_dict = {f"{model_engine_key}_oof": all_oof_probs}
    all_test_preds_dict = {f"{model_engine_key}_fold_{i}": probs for i, probs in enumerate(fold_test_probs_list)}
    
    # Generate final sequences using decoded ensemble logits
    final_test_sequences = ensemble(all_val_preds_dict, all_test_preds_dict, y_train_all)

    # 6. Calculate Prediction Statistics for Return Dict
    print("Step 5: Computing prediction distribution statistics...")
    
    def calculate_stats(probs_list: list) -> dict:
        if not probs_list or any(p is None for p in probs_list):
            return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
        # Flatten all probability frames across all sequences
        all_flattened = np.concatenate(probs_list, axis=0)
        return {
            "mean": float(np.mean(all_flattened)),
            "std": float(np.std(all_flattened)),
            "min": float(np.min(all_flattened)),
            "max": float(np.max(all_flattened))
        }

    # Average test probabilities across all folds for stats calculation
    num_test_samples = len(X_test_all)
    averaged_test_probs = []
    for i in range(num_test_samples):
        sample_fold_probs = [fold_test_probs_list[f][i] for f in range(n_splits)]
        averaged_test_probs.append(np.mean(sample_fold_probs, axis=0))

    prediction_stats = {
        "oof": calculate_stats(all_oof_probs),
        "test": calculate_stats(averaged_test_probs)
    }

    # 7. Generate Final Submission CSV
    print("Step 6: Writing final submission file...")
    submission_path = os.path.join(OUTPUT_DATA_PATH, "submission.csv")
    
    # Requirement: Columns 'Id,Sequence', Id is last 4 digits, Sequence is space-separated
    submission_df = pd.DataFrame({
        "Id": test_ids_all,
        "Sequence": final_test_sequences
    })
    
    submission_df.to_csv(submission_path, index=False)
    print(f"Final submission saved to {submission_path}")

    # 8. Return Deliverables
    output_info = {
        "submission_file_path": submission_path,
        "prediction_stats": prediction_stats,
        "pipeline_metadata": {
            "num_train": len(X_train_all),
            "num_test": len(X_test_all),
            "folds": n_splits,
            "engine": model_engine_key
        }
    }
    
    print("Workflow orchestration complete.")
    return output_info

if __name__ == "__main__":
    workflow()