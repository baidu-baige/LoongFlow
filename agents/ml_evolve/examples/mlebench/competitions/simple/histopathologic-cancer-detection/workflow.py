# -*- coding: utf-8 -*-
"""
LLM Generated Code
"""

import gc
import os

import numpy as np
import torch

from create_features import create_features
from cross_validation import cross_validation
from ensemble import ensemble
from load_data import load_data
from train_and_predict import PREDICTION_ENGINES

BASE_DATA_PATH = "/root/workspace/evolux_ml/output/mlebench/histopathologic-cancer-detection/prepared/public"
OUTPUT_DATA_PATH = "output/ee5d67f1-b3aa-447c-b45e-155ce5c4f09c/1/executor/output"


def workflow() -> dict:
    """
    Orchestrates the complete end-to-end machine learning pipeline in production mode.
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(OUTPUT_DATA_PATH):
        os.makedirs(OUTPUT_DATA_PATH, exist_ok=True)

    # 1. Load full dataset
    X, y, X_test, test_ids = load_data(validation_mode=False)

    # 2. Set up cross-validation strategy
    cv_splitter = cross_validation(X, y)

    # Initialize storage for predictions
    # all_oof_preds: {model_name: np.array of all training samples}
    # all_test_preds: {model_name: [fold_1_test_preds, fold_2_test_preds, ...]}
    all_oof_preds = {model_name: np.zeros(len(X)) for model_name in PREDICTION_ENGINES.keys()}
    all_test_preds = {model_name: [] for model_name in PREDICTION_ENGINES.keys()}

    # 3. Cross-validation loop
    for fold_idx, (train_idx, val_idx) in enumerate(cv_splitter.split(X, y)):
        X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
        y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]

        # a. Apply feature engineering for this fold
        X_tr_feat, y_tr_feat, X_val_feat, y_val_feat, X_te_feat = create_features(
            X_train_fold, y_train_fold, X_val_fold, y_val_fold, X_test
        )

        # b. Train each model in the registry
        for model_name, train_func in PREDICTION_ENGINES.items():
            # Train and get predictions
            val_preds, test_preds = train_func(
                X_tr_feat, y_tr_feat, X_val_feat, y_val_feat, X_te_feat
            )

            # Store OOF predictions using the validation indices
            all_oof_preds[model_name][val_idx] = val_preds
            # Store test predictions for this fold
            all_test_preds[model_name].append(test_preds)

            # Clear GPU memory
            torch.cuda.empty_cache()
            gc.collect()

    # 4. Ensemble predictions from all models and folds
    # y_true_full is the original labels y, which align with all_oof_preds filled by indices
    submission_df = ensemble(all_oof_preds, all_test_preds, y)

    # 5. Save artifacts
    submission_file_path = os.path.join(OUTPUT_DATA_PATH, "submission.csv")
    submission_df.to_csv(submission_file_path, index=False)

    # Save OOF predictions for potential future use (stacking)
    for model_name, oof in all_oof_preds.items():
        oof_path = os.path.join(OUTPUT_DATA_PATH, f"oof_{model_name}.npy")
        np.save(oof_path, oof)

    # 6. Generate deliverables
    # Calculate simple CV scores for information
    from sklearn.metrics import roc_auc_score
    model_scores = {}
    for model_name, oof in all_oof_preds.items():
        score = roc_auc_score(y, oof)
        model_scores[model_name] = float(score)

    output_info = {
        "submission_file_path": submission_file_path,
        "model_scores": model_scores,
        "num_folds": cv_splitter.get_n_splits(),
        "trained_models": list(PREDICTION_ENGINES.keys())
    }

    return output_info
