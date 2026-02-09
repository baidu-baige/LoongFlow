import os
import numpy as np
import pandas as pd
from typing import Dict

# Import all component functions
from load_data import load_data
from preprocess import preprocess
from get_splitter import get_splitter
from train_and_predict import PREDICTION_ENGINES
from ensemble import ensemble

BASE_DATA_PATH = "/mnt/pfs/loongflow/devmachine/2-05/evolux/output/mlebench/detecting-insults-in-social-commentary/prepared/public"
OUTPUT_DATA_PATH = "output/f0abd8d6-b251-4e86-b7b3-c7603506ee1b/1/executor/output"


def workflow() -> dict:
    """
    Orchestrates the complete end-to-end machine learning pipeline in production mode.
    Integrated components: load_data, preprocess, get_splitter, train_and_predict, and ensemble.
    """
    print("Workflow: Starting production pipeline execution...")

    # 1. Load complete dataset
    X_train_raw, y_train_raw, X_test_raw, test_ids = load_data(validation_mode=False)

    # 2. Initialize data splitting strategy (StratifiedKFold)
    splitter = get_splitter(X_train_raw, y_train_raw)
    n_folds = splitter.get_n_splits(X_train_raw, y_train_raw)

    # 3. Preparation for Fold-wise cross-validation
    # We use the ensemble engine defined in train_and_predict
    engine_name = "deberta_ridge_ensemble"
    train_fn = PREDICTION_ENGINES[engine_name]

    # Storage for Out-of-Fold and Test predictions
    oof_preds = np.zeros(len(y_train_raw), dtype=np.float32)
    test_preds_accum = np.zeros(len(X_test_raw), dtype=np.float32)

    print(f"Workflow: Beginning {n_folds}-fold cross-validation...")

    # Iterate through folds
    for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(X_train_raw, y_train_raw)):
        print(f"\n--- Fold {fold_idx + 1} / {n_folds} ---")

        # Split folds
        X_tr, X_val = X_train_raw.iloc[train_idx], X_train_raw.iloc[val_idx]
        y_tr, y_val = y_train_raw.iloc[train_idx], y_train_raw.iloc[val_idx]

        # Apply preprocessing for this specific fold
        # (Fit TF-IDF and Tokenizer logic on X_tr, transform X_val and X_test)
        X_tr_p, y_tr_p, X_val_p, y_val_p, X_test_p = preprocess(
            X_tr, y_tr, X_val, y_val, X_test_raw
        )

        # Train engine and generate predictions
        # train_fn returns (val_probs, test_probs)
        fold_val_preds, fold_test_preds = train_fn(
            X_tr_p, y_tr_p, X_val_p, y_val_p, X_test_p
        )

        # Collect OOF predictions and aggregate test predictions (averaging)
        oof_preds[val_idx] = fold_val_preds
        test_preds_accum += fold_test_preds / n_folds

    print("\nWorkflow: Cross-validation complete.")

    # 4. Ensemble Predictions
    # Although we have one "engine", the ensemble stage handles rank-averaging
    # and normalization to maximize AUC as per the competition requirements.
    # The ensemble function expects a dictionary mapping model names to predictions.
    all_val_preds = {engine_name: oof_preds}
    all_test_preds = {engine_name: test_preds_accum}

    final_test_probs = ensemble(
        all_val_preds=all_val_preds,
        all_test_preds=all_test_preds,
        y_val=y_train_raw
    )

    # 5. Compute Prediction Statistics
    prediction_stats = {
        "oof": {
            "mean": float(np.mean(oof_preds)),
            "std": float(np.std(oof_preds)),
            "min": float(np.min(oof_preds)),
            "max": float(np.max(oof_preds)),
        },
        "test": {
            "mean": float(np.mean(final_test_probs)),
            "std": float(np.std(final_test_probs)),
            "min": float(np.min(final_test_probs)),
            "max": float(np.max(final_test_probs)),
        }
    }

    # 6. Generate Deliverables and Save Artifacts
    os.makedirs(OUTPUT_DATA_PATH, exist_ok=True)

    # Save OOF predictions to disk for potential late-stage meta-ensembling
    oof_file_path = os.path.join(OUTPUT_DATA_PATH, "oof_predictions.npy")
    np.save(oof_file_path, oof_preds)

    # Generate final submission file
    # Based on the technical specification and file size analysis,
    # we use the sample submission structure if available, replacing the first column.
    sample_sub_path = os.path.join(BASE_DATA_PATH, "sample_submission_null.csv")
    submission_path = os.path.join(OUTPUT_DATA_PATH, "submission.csv")

    if os.path.exists(sample_sub_path):
        # Load template and overwrite the first column (Insult probability)
        df_submission = pd.read_csv(sample_sub_path)
        df_submission.iloc[:, 0] = final_test_probs
        df_submission.to_csv(submission_path, index=False)
    else:
        # Fallback to single-column format if template is missing
        df_submission = pd.DataFrame(final_test_probs)
        df_submission.to_csv(submission_path, index=False, header=False)

    print(f"Workflow: Final artifacts saved to {OUTPUT_DATA_PATH}")

    # 7. Return production summary
    return {
        "submission_file_path": submission_path,
        "prediction_stats": prediction_stats,
        "oof_file_path": oof_file_path,
        "model_engine": engine_name,
        "n_folds": n_folds
    }