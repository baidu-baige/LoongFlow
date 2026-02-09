import os
import numpy as np
import pandas as pd
from typing import Dict, List, Any

# Component Imports
from load_data import load_data
from preprocess import preprocess
from get_splitter import get_splitter
from train_and_predict import PREDICTION_ENGINES
from ensemble import ensemble

# Global paths based on environment specification
BASE_DATA_PATH = "/mnt/pfs/loongflow/devmachine/h20-10/evolux/output/mlebench/mlsp-2013-birds/prepared/public"
OUTPUT_DATA_PATH = "output/9f7a14b2-9e2e-4beb-a8af-238199431c62/57/executor/output"

def workflow() -> dict:
    """
    Orchestrates the complete end-to-end machine learning pipeline in production mode.

    This function integrates all pipeline components (data loading, preprocessing, 
    data splitting, model training, and ensembling) to generate final deliverables 
    specified in the task description.
    """
    # 0. Initialize environment
    print("\nStage 0: Initializing workflow environment...")
    os.makedirs(OUTPUT_DATA_PATH, exist_ok=True)

    # 1. Load full dataset
    # validation_mode=False ensures we process the complete set of 645 recordings (322 train, 323 test)
    print("Stage 1: Loading complete dataset (production mode)...")
    X_train_full, y_train_full, X_test_full, test_ids = load_data(validation_mode=False)
    
    # 2. Set up data splitting strategy
    # Uses MultilabelStratifiedKFold to handle sparse multi-label distributions
    print("Stage 2: Initializing multi-label iterative stratified splitter...")
    splitter = get_splitter(X_train_full, y_train_full)
    n_splits = splitter.get_n_splits()
    
    # 3. K-fold loop for training and validation
    print(f"Stage 3/4: Executing {n_splits}-fold CV with Per-Species Pseudo-Label Augmented Ensemble...")
    oof_preds = np.zeros(y_train_full.shape, dtype=np.float32)
    all_test_preds: Dict[str, np.ndarray] = {}
    
    # Select the specialized Per-Species Pseudo-Label Augmented Ensemble engine from registry
    model_fn = PREDICTION_ENGINES["per_species_pseudo_triple_ensemble"]
    
    for fold, (train_idx, val_idx) in enumerate(splitter.split(X_train_full, y_train_full)):
        print(f"\n--- Processing Fold {fold + 1}/{n_splits} ---")
        
        # a. Split raw multi-modal data
        X_tr = [X_train_full[i] for i in train_idx]
        X_va = [X_train_full[i] for i in val_idx]
        y_tr = y_train_full[train_idx]
        y_va = y_train_full[val_idx]
        
        # b. Preprocess
        # Extracts 572D features and prunes to 250D using RFE in parallel (36 cores)
        X_tr_p, y_tr_p, X_va_p, y_va_p, X_te_p = preprocess(X_tr, y_tr, X_va, y_va, X_test_full)
        
        # c. Train model and generate fold predictions
        # Includes Stage A (base training), Stage B (pseudo-labeling), and Stage C (per-species refinement)
        va_preds, te_preds = model_fn(X_tr_p, y_tr_p, X_va_p, y_va_p, X_te_p)
        
        # Store fold-specific results
        oof_preds[val_idx] = va_preds
        all_test_preds[f"fold_{fold}"] = te_preds
        
    # 4. Ensemble predictions from all folds
    print("\nStage 5: Aggregating fold predictions via ensembling...")
    # Map final OOF predictions for diagnostic AUC calculation within ensemble()
    all_val_preds = {k: oof_preds for k in all_test_preds.keys()}
    final_test_preds = ensemble(all_val_preds, all_test_preds, y_train_full)
    
    # 5. Compute prediction statistics for reporting
    print("Stage 6: Computing finalized prediction statistics...")
    prediction_stats = {
        "oof": {
            "mean": float(np.mean(oof_preds)),
            "std": float(np.std(oof_preds)),
            "min": float(np.min(oof_preds)),
            "max": float(np.max(oof_preds)),
        },
        "test": {
            "mean": float(np.mean(final_test_preds)),
            "std": float(np.std(final_test_preds)),
            "min": float(np.min(final_test_preds)),
            "max": float(np.max(final_test_preds)),
        }
    }
    
    # 6. Generate submission file
    # Required format: Id,Probability where Id = rec_id * 100 + species_idx
    # The submission must contain exactly 6138 lines (including header) for 323 test recordings
    print("Stage 7: Formatting submission and verifying integrity...")
    num_test_recordings = len(test_ids)
    num_species = 19
    
    submission_rows = []
    for i in range(num_test_recordings):
        rec_id = int(test_ids[i])
        for species_idx in range(num_species):
            # Combined Id calculation as per specification: rec_id * 100 + species_idx
            combined_id = int(rec_id * 100 + species_idx)
            # Ensure probabilities are in [0, 1] range
            probability = float(np.clip(final_test_preds[i, species_idx], 0.0, 1.0))
            submission_rows.append({"Id": combined_id, "Probability": probability})
    
    submission_df = pd.DataFrame(submission_rows)
    submission_file_path = os.path.join(OUTPUT_DATA_PATH, "submission.csv")
    submission_df.to_csv(submission_file_path, index=False)
    
    # Final compliance check
    expected_rows = num_test_recordings * num_species
    actual_rows = len(submission_df)
    if actual_rows != expected_rows:
        raise ValueError(f"CRITICAL: Row count mismatch. Expected {expected_rows}, got {actual_rows}.")
    else:
        print(f"Integrity check passed: {actual_rows + 1} total lines (including header) generated.")

    print(f"Workflow execution successful. Submission saved at: {submission_file_path}")

    return {
        "submission_file_path": submission_file_path,
        "prediction_stats": prediction_stats,
        "n_folds": int(n_splits),
        "total_test_recordings": int(num_test_recordings),
        "total_submission_rows": int(actual_rows),
    }

if __name__ == "__main__":
    workflow()