import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# Import all component functions
from load_data import load_data
from preprocess import preprocess
from get_splitter import get_splitter
from train_and_predict import PREDICTION_ENGINES
from ensemble import ensemble

BASE_DATA_PATH = "/mnt/pfs/loongflow/devmachine/2-05/evolux/output/mlebench/google-quest-challenge/prepared/public"
OUTPUT_DATA_PATH = "output/aaa741b3-cb02-44fc-a666-dd434e563444/8/executor/output"

def workflow() -> dict:
    """
    Orchestrates the complete end-to-end machine learning pipeline in production mode.
    Executes a 5-fold cross-validation using a dual-transformer (RoBERTa + BERT) ensemble.
    """
    print("Execution Stage: workflow")

    # 1. Load full dataset
    # Production mode: validation_mode=False processes the complete dataset.
    X_train_full, y_train_full, X_test, test_ids = load_data(validation_mode=False)
    
    # 2. Set up data splitting strategy
    # GroupKFold ensures questions are not split between train and validation.
    splitter = get_splitter(X_train_full, y_train_full)
    
    # Initialize containers for out-of-fold (OOF) and fold-wise test predictions
    n_train = len(X_train_full)
    n_targets = y_train_full.shape[1]
    target_columns = y_train_full.columns.tolist()
    
    oof_preds = np.zeros((n_train, n_targets), dtype=np.float32)
    all_test_preds_dict = {}
    
    # 3. Execute 5-fold Cross-Validation
    # Each fold involves training both RoBERTa and BERT backbones via the dual_transformer engine.
    print("Starting 5-fold cross-validation pipeline...")
    fold_iterator = splitter.split(X_train_full, y_train_full)
    
    # Identify the dual_transformer engine from the registry
    engine_key = "dual_transformer"
    if engine_key not in PREDICTION_ENGINES:
        raise KeyError(f"Critical Error: Prediction engine '{engine_key}' not found in registry.")
    train_fn = PREDICTION_ENGINES[engine_key]

    for fold_idx, (train_idx, val_idx) in enumerate(fold_iterator):
        print(f"\n--- Processing Fold {fold_idx + 1} / 5 ---")
        
        # a. Split train/validation data for this fold
        X_tr, X_val = X_train_full.iloc[train_idx].reset_index(drop=True), X_train_full.iloc[val_idx].reset_index(drop=True)
        y_tr, y_val = y_train_full.iloc[train_idx].reset_index(drop=True), y_train_full.iloc[val_idx].reset_index(drop=True)
        
        # b. Apply preprocess() to this fold
        # Returns model-ready Tensors/DataFrames for both RoBERTa and BERT structures.
        X_tr_p, y_tr_p, X_val_p, y_val_p, X_test_p = preprocess(X_tr, y_tr, X_val, y_val, X_test)
        
        # c. Train dual-model architecture and collect predictions
        # val_preds and test_preds are the averaged results of RoBERTa and BERT for this fold.
        val_preds, test_preds = train_fn(X_tr_p, y_tr_p, X_val_p, y_val_p, X_test_p)
        
        # Store results for global OOF and test ensemble
        oof_preds[val_idx] = val_preds
        all_test_preds_dict[f"dual_transformer_fold_{fold_idx}"] = test_preds
        
        print(f"Fold {fold_idx + 1} complete. OOF coverage: {np.count_nonzero(oof_preds.sum(axis=1))}/{n_train}")

    # 4. Ensemble predictions from all folds
    # The ensemble function handles arithmetic averaging, Min-Max scaling, and clipping.
    print("Merging fold predictions and applying post-processing...")
    all_val_preds_for_ensemble = {"global_oof": oof_preds}
    final_test_preds = ensemble(all_val_preds_for_ensemble, all_test_preds_dict, y_train_full)

    # 5. Compute global performance metrics
    print("Calculating final Mean Column-wise Spearman Correlation...")
    column_spearmans = []
    y_true_np = y_train_full.values
    for i in range(n_targets):
        with np.errstate(divide='ignore', invalid='ignore'):
            rho = spearmanr(y_true_np[:, i], oof_preds[:, i]).correlation
        if np.isnan(rho):
            rho = 0.0
        column_spearmans.append(rho)
    
    mean_cv_spearman = np.mean(column_spearmans)
    print(f"Global OOF Mean Spearman: {mean_cv_spearman:.4f}")

    # 6. Prepare deliverables and statistics
    os.makedirs(OUTPUT_DATA_PATH, exist_ok=True)
    
    # Save final submission file following the required format
    submission_df = pd.DataFrame(final_test_preds, columns=target_columns)
    submission_df.insert(0, 'qa_id', test_ids.values)
    submission_path = os.path.join(OUTPUT_DATA_PATH, "submission.csv")
    submission_df.to_csv(submission_path, index=False)
    
    # Save OOF predictions for downstream validation or stacking
    oof_path = os.path.join(OUTPUT_DATA_PATH, "oof_predictions.npy")
    np.save(oof_path, oof_preds)
    
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

    print(f"Workflow execution successful. Deliverables saved to {OUTPUT_DATA_PATH}")
    
    return {
        "submission_file_path": submission_path,
        "prediction_stats": prediction_stats,
        "oof_file_path": oof_path,
        "cv_mean_spearman": float(mean_cv_spearman)
    }