import os
import pandas as pd
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

# import all component functions
from load_data import load_data
from preprocess import preprocess
from get_splitter import get_splitter
from train_and_predict import PREDICTION_ENGINES
from ensemble import ensemble

BASE_DATA_PATH = "/mnt/pfs/loongflow/devmachine/h20-03/evolux/output/mlebench/aerial-cactus-identification/prepared/public"
OUTPUT_DATA_PATH = "output/eac64466-fa75-4fb8-ade1-75e2091ddf4a/1/executor/output"


def workflow() -> dict:
    """
    Orchestrates the complete end-to-end machine learning pipeline in production mode.
    """
    print("Starting pipeline: Aerial Cactus Identification")

    # Ensure output directory exists
    os.makedirs(OUTPUT_DATA_PATH, exist_ok=True)

    # 1. Load full dataset
    print("Step 1: Loading full dataset...")
    X_train_paths, y_train_labels, X_test_paths, test_ids = load_data(validation_mode=False)

    # 2. Set up data splitting strategy
    print("Step 2: Initializing data splitter (5-Fold Stratified)...")
    splitter = get_splitter(X_train_paths, y_train_labels)

    # Initialize containers for out-of-fold and test predictions
    oof_preds = np.zeros(len(X_train_paths))
    all_test_preds = {}

    # 3. Iterative preprocessing and training for each fold
    print("Step 3: Starting 5-Fold Cross-Validation loop...")
    train_fn = PREDICTION_ENGINES["densenet121_tta"]

    for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(X_train_paths, y_train_labels)):
        print(f"\n--- Processing Fold {fold_idx + 1}/5 ---")

        # Split paths and labels for this fold
        X_tr_paths = X_train_paths.iloc[train_idx]
        y_tr_labels = y_train_labels.iloc[train_idx]
        X_va_paths = X_train_paths.iloc[val_idx]
        y_va_labels = y_train_labels.iloc[val_idx]

        # Apply preprocessing
        print(f"Preprocessing images for Fold {fold_idx + 1}...")
        X_tr_p, y_tr_p, X_va_p, y_va_p, X_te_p = preprocess(
            X_tr_paths, y_tr_labels, X_va_paths, y_va_labels, X_test_paths
        )

        # Train model and generate predictions
        print(f"Training DenseNet121 with TTA for Fold {fold_idx + 1}...")
        val_p, test_p = train_fn(X_tr_p, y_tr_p, X_va_p, y_va_p, X_te_p)

        # Store predictions
        oof_preds[val_idx] = val_p
        all_test_preds[f"densenet_fold_{fold_idx}"] = test_p

        # Memory Management: Clear GPU cache and delete large tensor objects
        del X_tr_p, y_tr_p, X_va_p, y_va_p, X_te_p
        torch.cuda.empty_cache()
        print(f"Fold {fold_idx + 1} complete. CUDA cache cleared.")

    # 4. Ensemble predictions from all folds
    print("\nStep 4: Ensembling predictions...")
    # Prepare inputs for the ensemble function
    # Note: We provide the full y_train as y_val to calculate global OOF AUC in ensemble.py
    y_train_tensor = torch.tensor(y_train_labels.values, dtype=torch.float32)
    all_val_preds_for_ensemble = {"densenet121_tta_oof": oof_preds}

    final_test_preds = ensemble(all_val_preds_for_ensemble, all_test_preds, y_train_tensor)

    # 5. Compute performance and statistics
    cv_score = float(roc_auc_score(y_train_labels, oof_preds))
    print(f"Final Cross-Validation ROC AUC Score: {cv_score:.5f}")

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

    # 6. Generate deliverables
    print("Step 6: Saving artifacts and generating submission file...")
    submission_df = pd.DataFrame({
        "id": test_ids,
        "has_cactus": final_test_preds
    })

    submission_path = os.path.join(OUTPUT_DATA_PATH, "submission.csv")
    submission_df.to_csv(submission_path, index=False)

    output_info = {
        "submission_file_path": submission_path,
        "model_scores": {"densenet121_tta_cv": cv_score},
        "prediction_stats": prediction_stats,
    }

    print(f"Workflow execution finished. Results saved to {OUTPUT_DATA_PATH}")
    return output_info