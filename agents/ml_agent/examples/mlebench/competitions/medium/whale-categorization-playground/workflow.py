import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Subset
from load_data import load_data
from preprocess import preprocess
from get_splitter import get_splitter
from train_and_predict import PREDICTION_ENGINES
from ensemble import ensemble

BASE_DATA_PATH = "/mnt/pfs/loongflow/devmachine/2-05/evolux/output/mlebench/whale-categorization-playground/prepared/public"
OUTPUT_DATA_PATH = "output/47f6faee-7720-465c-b70e-dea18a3af55d/2/executor/output"

def workflow() -> dict:
    """
    Orchestrates the complete end-to-end machine learning pipeline in production mode.
    
    Addresses the 'unseen labels' error by ensuring every unique whale ID is present 
    in the training split of every fold, thus maintaining label consistency for 
    the immutable preprocess and ensemble components.
    """
    if not os.path.exists(OUTPUT_DATA_PATH):
        os.makedirs(OUTPUT_DATA_PATH, exist_ok=True)

    # 1. Load full dataset
    print("Loading full dataset for production...")
    X_train_full, y_train_full, X_test_full, test_ids = load_data(validation_mode=False)

    # 2. Get splitting strategy (5-Fold Stratified)
    print("Setting up cross-validation splits...")
    splitter = get_splitter(X_train_full, y_train_full)
    
    # Identify indices of the first occurrence of every class.
    # We must ensure these are in the training set of every fold because the 
    # 'preprocess' function fits its LabelEncoder strictly on the 'y_train' argument.
    # Classes with only one sample would otherwise cause a KeyError when present in validation.
    must_have_indices = y_train_full.index[~y_train_full.duplicated(keep='first')].values
    all_indices = np.arange(len(y_train_full))
    
    all_val_preds_dict = {}
    all_test_preds_dict = {}
    all_y_val_processed = []

    # 3. K-Fold Training and Inference Loop
    model_fn = PREDICTION_ENGINES["convnext_arcface"]
    
    for fold, (raw_train_idx, raw_val_idx) in enumerate(splitter.split(X_train_full, y_train_full)):
        print(f"--- Executing Fold {fold + 1}/5 ---")
        
        # Refine indices to ensure no class is 'unseen' by the fold's LabelEncoder
        # Move any sample that is the sole representative of its class from val to train
        fold_val_idx = np.array([i for i in raw_val_idx if i not in must_have_indices])
        fold_train_idx = np.array([i for i in all_indices if i not in fold_val_idx])
        
        # Create Data Subsets
        X_train_fold = Subset(X_train_full, fold_train_idx)
        y_train_fold = y_train_full.iloc[fold_train_idx]
        X_val_fold = Subset(X_train_full, fold_val_idx)
        y_val_fold = y_train_full.iloc[fold_val_idx]
        
        # Preprocess: LabelEncoder fit happens here on y_train_fold
        print(f"Preprocessing images and labels for Fold {fold + 1}...")
        (X_tr_p, y_tr_p, 
         X_val_p, y_val_p, 
         X_te_p) = preprocess(X_train_fold, y_train_fold, X_val_fold, y_val_fold, X_test_full)
        
        # Train and Predict using DDP on available GPUs
        print(f"Training ConvNeXt-ArcFace on Fold {fold + 1}...")
        val_preds, test_preds = model_fn(X_tr_p, y_tr_p, X_val_p, y_val_p, X_te_p)
        
        # Store results for ensembling
        # Use keys that sort correctly for ensemble's OOF reconstruction
        all_val_preds_dict[f"fold_{fold:02d}"] = val_preds
        all_test_preds_dict[f"fold_{fold:02d}"] = test_preds
        all_y_val_processed.append(y_val_p)
        
        torch.cuda.empty_cache()

    # 4. Concatenate validation targets for threshold optimization
    y_val_all = torch.cat(all_y_val_processed, dim=0)

    # 5. Ensemble and Post-process
    print("Ensembling fold predictions and optimizing 'new_whale' threshold...")
    final_test_predictions = ensemble(all_val_preds_dict, all_test_preds_dict, y_val_all)

    # 6. Compute prediction statistics for reporting
    # Calculate stats on combined OOF logits and averaged test logits
    oof_logits = np.concatenate([all_val_preds_dict[k] for k in sorted(all_val_preds_dict.keys())], axis=0)
    avg_test_logits = np.mean(list(all_test_preds_dict.values()), axis=0)

    prediction_stats = {
        "oof": {
            "mean": float(np.mean(oof_logits)),
            "std": float(np.std(oof_logits)),
            "min": float(np.min(oof_logits)),
            "max": float(np.max(oof_logits)),
        },
        "test": {
            "mean": float(np.mean(avg_test_logits)),
            "std": float(np.std(avg_test_logits)),
            "min": float(np.min(avg_test_logits)),
            "max": float(np.max(avg_test_logits)),
        }
    }

    # 7. Generate Deliverables
    submission_df = pd.DataFrame({
        'Image': test_ids,
        'Id': final_test_predictions
    })
    
    submission_path = os.path.join(OUTPUT_DATA_PATH, "submission.csv")
    submission_df.to_csv(submission_path, index=False)
    
    print(f"Pipeline execution complete. Final submission file at: {submission_path}")

    return {
        "submission_file_path": submission_path,
        "prediction_stats": prediction_stats
    }