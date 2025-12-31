# -*- coding: utf-8 -*-
"""
LLM Generated Code
"""

import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from create_features import create_features
from cross_validation import cross_validation
from ensemble import ensemble
from load_data import load_data
from train_and_predict import PREDICTION_ENGINES

BASE_DATA_PATH = "/root/workspace/evolux/output/mlebench/plant-pathology-2020-fgvc7/prepared/public"
OUTPUT_DATA_PATH = "output/1799683e-18f2-43a3-97d1-8b0bdddc3200/1/executor/output"


def workflow() -> dict:
    """
    Orchestrates the complete end-to-end machine learning pipeline in production mode.
    """
    # 1. Load full dataset
    X, y, X_test, test_ids = load_data(validation_mode=False)

    # 2. Set up cross-validation strategy
    splitter = cross_validation(X, y)

    # 3. Initialize prediction storage
    # all_oof_preds stores full OOF predictions for each model
    # all_test_preds stores a list of test set predictions from each fold
    all_oof_preds = {}
    all_test_preds = {}
    for model_name in PREDICTION_ENGINES.keys():
        all_oof_preds[model_name] = np.zeros((len(X), y.shape[1]))
        all_test_preds[model_name] = []

    # 4. Cross-validation loop
    for fold, (train_idx, val_idx) in enumerate(splitter.split(X, y)):
        X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
        y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]

        # Apply create_features to this fold
        X_tr, y_tr, X_va, y_va, X_te = create_features(
            X_train_fold, y_train_fold, X_val_fold, y_val_fold, X_test
        )

        # Train each model available in the registry
        for model_name, train_fn in PREDICTION_ENGINES.items():
            # Execute training and obtain predictions
            val_preds, test_preds = train_fn(X_tr, y_tr, X_va, y_va, X_te)

            # Store OOF predictions and test fold predictions
            all_oof_preds[model_name][val_idx] = val_preds
            all_test_preds[model_name].append(test_preds)

    # 5. Ensemble predictions from all models and folds
    # Pass full labels for any potential optimization/evaluation within ensemble
    final_test_preds = ensemble(all_oof_preds, all_test_preds, y)

    # 6. Calculate validation scores
    model_scores = {}
    for model_name, oof_preds in all_oof_preds.items():
        # Evaluation is based on mean column-wise ROC AUC
        score = roc_auc_score(y.values, oof_preds, average='macro', multi_class='ovr')
        model_scores[model_name] = float(score)

    # 7. Generate final submission artifact
    # Construct DataFrame with required columns: image_id, healthy, multiple_diseases, rust, scab
    submission_df = pd.DataFrame(final_test_preds, columns=y.columns)
    submission_df.insert(0, 'image_id', test_ids.values)

    # Ensure output directory exists before saving
    if not os.path.exists(OUTPUT_DATA_PATH):
        os.makedirs(OUTPUT_DATA_PATH, exist_ok=True)

    submission_path = os.path.join(OUTPUT_DATA_PATH, "submission.csv")
    submission_df.to_csv(submission_path, index=False)

    # 8. Return task deliverables in a serializable format
    return {
        "submission_file_path": submission_path,
        "model_scores": model_scores
    }
