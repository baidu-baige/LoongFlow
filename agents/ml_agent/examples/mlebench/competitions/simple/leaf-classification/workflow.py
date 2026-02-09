import os
import numpy as np
import pandas as pd
import cudf
from typing import Dict, Tuple

# Import component functions
from load_data import load_data
from preprocess import preprocess
from get_splitter import get_splitter
from train_and_predict import PREDICTION_ENGINES
from ensemble import ensemble

BASE_DATA_PATH = "/mnt/pfs/loongflow/devmachine/h20-09/evolux/output/mlebench/leaf-classification/prepared/public"
OUTPUT_DATA_PATH = "output/5e63fe40-52af-4d8b-ac71-4d3a91b9999f/54/executor/output"

def _run_cv_pass(X_train: cudf.DataFrame, y_train: cudf.Series, X_test: cudf.DataFrame, num_classes: int, pass_name: str) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Executes a multi-fold cross-validation pass for all registered prediction engines.
    """
    splitter = get_splitter(X_train, y_train)
    all_oof: Dict[str, np.ndarray] = {}
    all_test: Dict[str, np.ndarray] = {}
    n_splits = splitter.get_n_splits()

    for engine_name, engine_fn in PREDICTION_ENGINES.items():
        print(f"Workflow [{pass_name}]: Training engine '{engine_name}'...")
        oof_preds = np.zeros((len(X_train), num_classes), dtype=np.float32)
        test_preds_accum = np.zeros((len(X_test), num_classes), dtype=np.float32)
        
        for fold, (train_idx, val_idx) in enumerate(splitter.split(X_train, y_train), 1):
            print(f"Workflow [{pass_name}]: Engine '{engine_name}' - Fold {fold}/{n_splits}")
            
            # Split and reset indices for fold-specific processing
            X_tr, X_val = X_train.iloc[train_idx].reset_index(drop=True), X_train.iloc[val_idx].reset_index(drop=True)
            y_tr, y_val = y_train.iloc[train_idx].reset_index(drop=True), y_train.iloc[val_idx].reset_index(drop=True)
            
            # Preprocess fold data
            X_tr_p, y_tr_p, X_val_p, y_val_p, X_test_p = preprocess(X_tr, y_tr, X_val, y_val, X_test)
            
            # Train and generate predictions
            v_p, t_p = engine_fn(X_tr_p, y_tr_p, X_val_p, y_val_p, X_test_p)
            
            # Map OOF predictions back to original indices
            if hasattr(val_idx, 'to_numpy'):
                idx_map = val_idx.to_numpy()
            elif hasattr(val_idx, 'get'):
                idx_map = val_idx.get()
            else:
                idx_map = np.asarray(val_idx)
                
            oof_preds[idx_map] = v_p
            test_preds_accum += t_p
            
        all_oof[engine_name] = oof_preds
        all_test[engine_name] = test_preds_accum / n_splits
        
    return all_oof, all_test

def workflow() -> dict:
    """
    Orchestrates the complete end-to-end machine learning pipeline with 3-Pass Iterative Pseudo-Labeling.
    """
    print("Workflow: Initializing 3-Pass Iterative Pseudo-Labeling Production Pipeline.")
    os.makedirs(OUTPUT_DATA_PATH, exist_ok=True)

    # 1. Load full dataset
    X_train_orig, y_train_orig, X_test_orig, test_ids = load_data(validation_mode=False)
    num_classes = 99

    # ==========================================
    # PASS 1: Initial Cross-Validation
    # ==========================================
    print("Workflow: Starting Pass 1 - Initial 9-fold CV...")
    all_oof_p1, all_test_p1 = _run_cv_pass(X_train_orig, y_train_orig, X_test_orig, num_classes, "Pass 1")
    
    # Pass 1 Ensemble
    p1_test_probs = ensemble(all_val_preds=all_oof_p1, all_test_preds=all_test_p1, y_val=y_train_orig)

    # ==========================================
    # PASS 2: First Iterative Augmentation (Threshold > 0.95)
    # ==========================================
    confidences_p1 = p1_test_probs.max(axis=1)
    pseudo_mask_p2 = confidences_p1 > 0.95
    num_pseudo_p2 = int(pseudo_mask_p2.sum())
    print(f"Workflow: Pass 2 - Identified {num_pseudo_p2} samples with confidence > 0.95 for augmentation.")

    if num_pseudo_p2 > 0:
        X_pseudo_p2 = X_test_orig.iloc[pseudo_mask_p2].reset_index(drop=True)
        y_pseudo_p2 = cudf.Series(np.argmax(p1_test_probs[pseudo_mask_p2], axis=1))
        
        X_train_aug1 = cudf.concat([X_train_orig, X_pseudo_p2], axis=0).reset_index(drop=True)
        y_train_aug1 = cudf.concat([y_train_orig, y_pseudo_p2], axis=0).reset_index(drop=True)
    else:
        print("Workflow Warning: Pass 2 - No samples met threshold. Using original training set.")
        X_train_aug1, y_train_aug1 = X_train_orig, y_train_orig

    print("Workflow: Starting Pass 2 - Training on first augmented dataset...")
    all_oof_p2, all_test_p2 = _run_cv_pass(X_train_aug1, y_train_aug1, X_test_orig, num_classes, "Pass 2")
    
    # Pass 2 Ensemble
    p2_test_probs = ensemble(all_val_preds=all_oof_p2, all_test_preds=all_test_p2, y_val=y_train_aug1)

    # ==========================================
    # PASS 3: Second Iterative Augmentation (Threshold > 0.90)
    # ==========================================
    confidences_p2 = p2_test_probs.max(axis=1)
    pseudo_mask_p3 = confidences_p2 > 0.90
    num_pseudo_p3 = int(pseudo_mask_p3.sum())
    print(f"Workflow: Pass 3 - Identified {num_pseudo_p3} samples with confidence > 0.90 for augmentation.")

    if num_pseudo_p3 > 0:
        X_pseudo_p3 = X_test_orig.iloc[pseudo_mask_p3].reset_index(drop=True)
        y_pseudo_p3 = cudf.Series(np.argmax(p2_test_probs[pseudo_mask_p3], axis=1))
        
        X_train_aug2 = cudf.concat([X_train_orig, X_pseudo_p3], axis=0).reset_index(drop=True)
        y_train_aug2 = cudf.concat([y_train_orig, y_pseudo_p3], axis=0).reset_index(drop=True)
    else:
        print("Workflow Warning: Pass 3 - No samples met threshold. Using Pass 2 training set.")
        X_train_aug2, y_train_aug2 = X_train_aug1, y_train_aug1

    print("Workflow: Starting Pass 3 - Final Training on second augmented dataset...")
    all_oof_p3, all_test_p3 = _run_cv_pass(X_train_aug2, y_train_aug2, X_test_orig, num_classes, "Pass 3")
    
    # Final Ensemble
    final_test_preds = ensemble(all_val_preds=all_oof_p3, all_test_preds=all_test_p3, y_val=y_train_aug2)

    # ==========================================
    # FINALIZATION AND OUTPUT
    # ==========================================
    print("Workflow: Applying final numerical stability normalization (clip=1e-7)...")
    final_test_preds = np.clip(final_test_preds, 1e-7, 1.0 - 1e-7)
    final_test_preds /= final_test_preds.sum(axis=1, keepdims=True)

    # Save submission artifact
    sample_sub_path = os.path.join(BASE_DATA_PATH, "sample_submission.csv")
    sample_sub = pd.read_csv(sample_sub_path)
    species_columns = sample_sub.columns[1:]
    
    submission_df = pd.DataFrame(final_test_preds, columns=species_columns)
    submission_df.insert(0, 'id', test_ids.to_pandas().values)
    
    submission_file_path = os.path.join(OUTPUT_DATA_PATH, "submission.csv")
    submission_df.to_csv(submission_file_path, index=False)
    
    # Calculate Prediction Statistics
    primary_engine = list(PREDICTION_ENGINES.keys())[0]
    final_oof = all_oof_p3[primary_engine]
    
    prediction_stats = {
        "oof": {
            "mean": float(np.mean(final_oof)),
            "std": float(np.std(final_oof)),
            "min": float(np.min(final_oof)),
            "max": float(np.max(final_oof)),
        },
        "test": {
            "mean": float(np.mean(final_test_preds)),
            "std": float(np.std(final_test_preds)),
            "min": float(np.min(final_test_preds)),
            "max": float(np.max(final_test_preds)),
        }
    }

    print(f"Workflow Complete. Final submission saved to {submission_file_path}")
    
    return {
        "submission_file_path": submission_file_path,
        "prediction_stats": prediction_stats,
        "pass2_pseudo_count": num_pseudo_p2,
        "pass3_pseudo_count": num_pseudo_p3
    }

if __name__ == "__main__":
    workflow()