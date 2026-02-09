import os
import pandas as pd
import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

# Import all component functions from the pipeline
from load_data import load_data
from preprocess import preprocess
from get_splitter import get_splitter
from train_and_predict import PREDICTION_ENGINES
from ensemble import ensemble

BASE_DATA_PATH = "/mnt/pfs/loongflow/devmachine/2-05/evolux/output/mlebench/random-acts-of-pizza/prepared/public"
OUTPUT_DATA_PATH = "output/f2dbb22d-a0cb-4add-aa87-f2c6b1a4b76f/77/executor/output"

def workflow() -> dict:
    """
    Orchestrates the complete end-to-end machine learning pipeline in production mode.
    Executes a 5-fold CV with integrated Pseudo-Labeling, Target Encoding, and a final 
    deterministic heuristic override for known givers.
    """
    # Step 0: Ensure output directory exists
    if not os.path.exists(OUTPUT_DATA_PATH):
        os.makedirs(OUTPUT_DATA_PATH, exist_ok=True)

    # Step 1: Load complete dataset (Production mode)
    print("Workflow: Loading full dataset...")
    X_train_raw, y_train_raw, X_test_raw, test_ids = load_data(validation_mode=False)

    # Step 2: Initialize data splitter (StratifiedKFold)
    print("Workflow: Initializing data splitter...")
    splitter = get_splitter(X_train_raw, y_train_raw)
    n_splits = splitter.get_n_splits()

    # Initialize containers for Out-of-Fold (OOF) and Test predictions
    # PREDICTION_ENGINES contains the 'hybrid_pseudo_ensemble' engine
    model_names = list(PREDICTION_ENGINES.keys())
    oof_preds_dict = {name: np.zeros(len(y_train_raw)) for name in model_names}
    test_preds_accum = {name: np.zeros(len(X_test_raw)) for name in model_names}

    # Step 3: Cross-Validation Loop
    print(f"Workflow: Starting {n_splits}-fold cross-validation loop...")
    for fold, (train_idx, val_idx) in enumerate(splitter.split(X_train_raw, y_train_raw)):
        print(f"\n--- Processing Fold {fold + 1}/{n_splits} ---")
        
        # Split raw data for this fold
        X_tr_fold, X_va_fold = X_train_raw.iloc[train_idx], X_train_raw.iloc[val_idx]
        y_tr_fold, y_va_fold = y_train_raw.iloc[train_idx], y_train_raw.iloc[val_idx]

        # a. Preprocess data (Fold-wise to ensure zero leakage in Target Encoding and PCA)
        X_tr_p, y_tr_p, X_va_p, y_va_p, X_te_p = preprocess(
            X_tr_fold, y_tr_fold, X_va_fold, y_va_fold, X_test_raw
        )

        # b. Train each prediction engine (Hybrid DeBERTa + Pseudo-Labeled Tabular Models)
        for name, train_fn in PREDICTION_ENGINES.items():
            print(f"Executing training engine: {name}")
            # train_fn returns (val_preds, test_preds) for this fold
            val_preds, test_preds = train_fn(X_tr_p, y_tr_p, X_va_p, y_va_p, X_te_p)
            
            # Record OOF predictions for this fold's validation set
            oof_preds_dict[name][val_idx] = val_preds
            # Accumulate test predictions for later averaging
            test_preds_accum[name] += test_preds / n_splits

    # Step 4: Ensemble Stage (Rank-Weighted Averaging)
    print("\nWorkflow: Executing Rank-Weighted Ensemble...")
    # This applies rank normalization and integrates the Penta-Engine results
    final_test_preds = ensemble(
        all_val_preds=oof_preds_dict,
        all_test_preds=test_preds_accum,
        y_val=y_train_raw
    )

    # Step 5: Deterministic Heuristic Override
    # "The 100% precision of giver_username_if_known != 'N/A' must be preserved"
    print("Workflow: Applying deterministic override for known givers...")
    if 'giver_username_if_known' in X_test_raw.columns:
        giver_mask = (X_test_raw['giver_username_if_known'] != 'N/A').values
        num_overrides = np.sum(giver_mask)
        final_test_preds[giver_mask] = 1.0
        print(f"Applied 1.0 probability override for {num_overrides} samples.")
    else:
        print("Warning: 'giver_username_if_known' not found in test features. Skipping override.")

    # Step 6: Compute Final Statistics
    print("Workflow: Computing prediction distribution statistics...")

    def get_rank_norm(preds: np.ndarray) -> np.ndarray:
        """Helper to replicate ensemble.py rank-normalization for OOF stats."""
        if len(preds) <= 1: return preds
        return (rankdata(preds, method='average') - 1) / (len(preds) - 1)

    # Replicate Rank-Weighted Averaging logic for OOF (Egalitarian fallback for engine names)
    final_oof_preds = np.zeros(len(y_train_raw))
    for name in oof_preds_dict.keys():
        norm_ranks = get_rank_norm(oof_preds_dict[name])
        final_oof_preds += (1.0 / len(oof_preds_dict)) * norm_ranks

    def calculate_stats(preds: np.ndarray) -> dict:
        return {
            "mean": float(np.mean(preds)),
            "std": float(np.std(preds)),
            "min": float(np.min(preds)),
            "max": float(np.max(preds)),
        }

    prediction_stats = {
        "oof": calculate_stats(final_oof_preds),
        "test": calculate_stats(final_test_preds)
    }
    
    # Calculate and print final OOF performance for monitoring
    ensemble_auc = roc_auc_score(y_train_raw, final_oof_preds)
    print(f"Final OOF Ensemble ROC-AUC: {ensemble_auc:.4f}")

    # Step 7: Generate Submission File
    print("Workflow: Generating final submission artifacts...")
    submission_df = pd.DataFrame({
        'request_id': test_ids.values,
        'requester_received_pizza': final_test_preds
    })
    
    submission_file_path = os.path.join(OUTPUT_DATA_PATH, "submission.csv")
    submission_df.to_csv(submission_file_path, index=False)
    print(f"Submission saved successfully to: {submission_file_path}")

    # Final Stage: Deliverables Collection
    output_info = {
        "submission_file_path": submission_file_path,
        "prediction_stats": prediction_stats,
    }

    print("Workflow execution completed successfully.")
    return output_info

if __name__ == "__main__":
    workflow()