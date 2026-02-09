import os
import pandas as pd
import numpy as np
from typing import Dict

# Import all component functions
from load_data import load_data
from preprocess import preprocess
from get_splitter import get_splitter
from train_and_predict import PREDICTION_ENGINES
from ensemble import ensemble

BASE_DATA_PATH = "/mnt/pfs/loongflow/devmachine/h20-09/evolux/output/mlebench/predict-volcanic-eruptions-ingv-oe/prepared/public"
OUTPUT_DATA_PATH = "output/bdc750a4-f0a3-4926-871d-f9675d7cf1ef/1/executor/output"

def workflow() -> dict:
    """
    Orchestrates the complete end-to-end machine learning pipeline in production mode.
    """
    print("Workflow: Starting production pipeline...")
    
    # Ensure output directory exists
    os.makedirs(OUTPUT_DATA_PATH, exist_ok=True)

    # 1. Load full dataset
    # Returns: X_train (dict), y_train (Series), X_test (dict), test_ids (ndarray)
    X_train_raw, y_train, X_test_raw, test_ids = load_data(validation_mode=False)

    # 2. Set up data splitting strategy
    splitter = get_splitter(X_train_raw, y_train)
    n_splits = splitter.get_n_splits()

    # Containers for out-of-fold and test predictions for each engine
    model_oof_preds = {}
    model_test_preds = {}

    # Initialize prediction containers for each registered engine
    for engine_name in PREDICTION_ENGINES.keys():
        model_oof_preds[engine_name] = np.zeros(len(y_train))
        model_test_preds[engine_name] = np.zeros(len(test_ids))

    # 3. Cross-Validation Loop
    print(f"Workflow: Starting {n_splits}-fold cross-validation...")
    for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(X_train_raw, y_train)):
        print(f"--- Processing Fold {fold_idx + 1}/{n_splits} ---")
        
        # Subset training and validation data for this fold
        # Convert indices to segment_ids for signal dictionary lookup
        train_ids_fold = y_train.index[train_idx]
        val_ids_fold = y_train.index[val_idx]
        
        X_tr_fold = {sid: X_train_raw[sid] for sid in train_ids_fold}
        y_tr_fold = y_train.iloc[train_idx]
        X_val_fold = {sid: X_train_raw[sid] for sid in val_ids_fold}
        y_val_fold = y_train.iloc[val_idx]

        # b. Preprocess: Feature extraction and scaling
        # Note: preprocess handles parallel extraction and fit/transform logic
        X_tr_proc, y_tr_proc, X_val_proc, y_val_proc, X_te_proc = preprocess(
            X_tr_fold, y_tr_fold, X_val_fold, y_val_fold, X_test_raw
        )

        # c. Train each model in PREDICTION_ENGINES
        for engine_name, engine_fn in PREDICTION_ENGINES.items():
            print(f"Workflow: Training engine '{engine_name}' on Fold {fold_idx + 1}...")
            # Each engine optimizes MAE and uses GPUs
            val_preds, test_preds = engine_fn(
                X_tr_proc, y_tr_proc, X_val_proc, y_val_proc, X_te_proc
            )
            
            # Record OOF predictions and accumulate test predictions (for averaging)
            model_oof_preds[engine_name][val_idx] = val_preds
            model_test_preds[engine_name] += test_preds / n_splits

    # 4. Ensemble stage: Consolidate model predictions
    # This applies blending weights (default 1/N) to the averaged test predictions
    final_test_preds = ensemble(model_oof_preds, model_test_preds, y_train)

    # Calculate final OOF predictions using the same blending logic (equal weights if 1 engine)
    # We use equal weighting as per technical spec logic to generate OOF stats
    model_names = list(PREDICTION_ENGINES.keys())
    weights = np.ones(len(model_names)) / len(model_names)
    final_oof_preds = np.zeros(len(y_train))
    for i, name in enumerate(model_names):
        final_oof_preds += weights[i] * model_oof_preds[name]

    # 5. Compute prediction statistics
    prediction_stats = {
        "oof": {
            "mean": float(np.mean(final_oof_preds)),
            "std": float(np.std(final_oof_preds)),
            "min": float(np.min(final_oof_preds)),
            "max": float(np.max(final_oof_preds)),
        },
        "test": {
            "mean": float(np.mean(final_test_preds)),
            "std": float(np.std(final_test_preds)),
            "min": float(np.min(final_test_preds)),
            "max": float(np.max(final_test_preds)),
        }
    }

    # 6. Generate deliverables
    submission_df = pd.DataFrame({
        'segment_id': test_ids,
        'time_to_eruption': final_test_preds
    })
    
    submission_path = os.path.join(OUTPUT_DATA_PATH, "submission.csv")
    submission_df.to_csv(submission_path, index=False)
    print(f"Workflow: Submission saved to {submission_path}")

    # Log overall CV performance
    overall_mae = np.mean(np.abs(final_oof_preds - y_train.values))
    print(f"Workflow: Final CV MAE: {overall_mae:.2f}")

    output_info = {
        "submission_file_path": submission_path,
        "prediction_stats": prediction_stats,
        "overall_cv_mae": float(overall_mae)
    }
    
    print("Workflow: Pipeline execution completed successfully.")
    return output_info