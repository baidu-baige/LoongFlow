# -*- coding: utf-8 -*-
"""
LLM Generated Code
"""

import os

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from create_features import create_features
from cross_validation import cross_validation
from ensemble import ensemble
from load_data import load_data
from train_and_predict import PREDICTION_ENGINES

BASE_DATA_PATH = "/root/workspace/evolux/output/mlebench/us-patent-phrase-to-phrase-matching/prepared/public"
OUTPUT_DATA_PATH = "output/aefcc010-8f21-4ecb-b149-7bf99579e6d3/6/executor/output"


def workflow() -> dict:
    """
    Orchestrates the complete end-to-end machine learning pipeline in production mode.
    
    This function integrates all pipeline components (data loading, feature engineering, 
    cross-validation, model training, and ensembling) to generate final deliverables 
    specified in the task description.
    """
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DATA_PATH, exist_ok=True)

    # 1. Load full dataset for production training
    X, y, X_test, test_ids = load_data(validation_mode=False)

    # 2. Set up cross-validation strategy
    # This call adds a 'fold' column to the dataframe X and returns the splitter
    _ = cross_validation(X, y)
    n_splits = 5  # Number of splits defined in cross_validation.py

    # 3. Initialize storage for predictions across different model architectures
    model_names = list(PREDICTION_ENGINES.keys())
    all_oof_preds = {name: np.zeros(len(X)) for name in model_names}
    all_test_preds = {name: [] for name in model_names}

    # 4. Iterate through each fold of the cross-validation
    for fold_idx in range(n_splits):
        # Split Data into Training and Validation sets for the current fold
        train_mask = X['fold'] != fold_idx
        val_mask = X['fold'] == fold_idx

        X_train_f = X[train_mask].copy()
        y_train_f = y[train_mask].copy()
        X_val_f = X[val_mask].copy()
        y_val_f = y[val_mask].copy()

        # Step 3a: Apply feature engineering (Tokenization, Sequence Construction)
        (X_train_trans, y_train_trans,
         X_val_trans, y_val_trans,
         X_test_trans) = create_features(X_train_f, y_train_f, X_val_f, y_val_f, X_test.copy())

        # Step 3b: Train each model in the registry and generate predictions
        for model_name in model_names:
            train_fn = PREDICTION_ENGINES[model_name]

            # This handles model initialization, training, and inference on Val/Test sets
            val_preds, test_preds = train_fn(
                X_train_trans, y_train_trans,
                X_val_trans, y_val_trans,
                X_test_trans
            )

            # Store Out-of-Fold predictions and Test set predictions
            all_oof_preds[model_name][val_mask] = val_preds
            all_test_preds[model_name].append(test_preds)

    # 5. Evaluate individual model performance using Pearson correlation
    model_scores = {}
    for model_name in model_names:
        score, _ = pearsonr(y, all_oof_preds[model_name])
        model_scores[model_name] = float(score)

    # 6. Ensemble predictions from all models and all folds
    # This applies arithmetic averaging across all individual fold predictions
    final_test_preds = ensemble(all_oof_preds, all_test_preds, y)

    # Calculate the global Ensembled OOF Pearson correlation score
    # We aggregate OOF predictions by simple averaging across model types
    ensemble_oof = np.mean([all_oof_preds[name] for name in model_names], axis=0)
    ensemble_cv_score, _ = pearsonr(y, ensemble_oof)

    # 7. Generate deliverables and save artifacts

    # Create submission.csv
    submission_df = pd.DataFrame({
        'id': test_ids,
        'score': final_test_preds
    })
    submission_file_path = os.path.join(OUTPUT_DATA_PATH, "submission.csv")
    submission_df.to_csv(submission_file_path, index=False)

    # Save OOF predictions for traceability
    oof_file_path = os.path.join(OUTPUT_DATA_PATH, "oof_predictions.csv")
    pd.DataFrame(all_oof_preds).to_csv(oof_file_path, index=False)

    # Construct the serializable output dictionary
    output_info = {
        "submission_file_path": submission_file_path,
        "oof_predictions_path": oof_file_path,
        "model_scores": model_scores,
        "ensemble_cv_score": float(ensemble_cv_score)
    }

    return output_info
