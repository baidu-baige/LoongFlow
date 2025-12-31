# -*- coding: utf-8 -*-
"""
LLM Generated Code
"""

import os

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from create_features import create_features
from cross_validation import cross_validation
from ensemble import ensemble
from load_data import load_data
from train_and_predict import PREDICTION_ENGINES

BASE_DATA_PATH = "/root/workspace/evolux_ml/output/mlebench/text-normalization-challenge-english-language/prepared/public"
OUTPUT_DATA_PATH = "output/12e29d80-a70c-426a-9331-de3aa1a6ce7c/14/executor/output"


def workflow() -> dict:
    """
    Orchestrates the complete end-to-end machine learning pipeline in production mode.
    """
    # 1. Load full dataset
    # X: ['sentence_id', 'token_id', 'before']
    # y: ['after', 'class']
    # X_test: ['sentence_id', 'token_id', 'before']
    # test_ids: Series of 'sentence_id_token_id'
    X, y, X_test, test_ids = load_data(validation_mode=False)

    # 2. Set up cross-validation strategy
    splitter = cross_validation(X, y)

    # Initialize containers for Out-of-Fold (OOF) and Test predictions
    # We store the intent class labels predicted by the model
    oof_class_preds = np.empty(len(X), dtype=object)
    test_preds_by_fold = []

    # Determine model configuration
    model_name = "lgbm_intent_classifier"
    train_fn = PREDICTION_ENGINES[model_name]

    # 3. Cross-Validation Loop
    fold_idx = 0
    # StratifiedGroupKFoldByClass groups by sentence_id and stratifies by class
    for train_idx, val_idx in splitter.split(X, y):
        # Split train/validation data for the current fold
        X_tr_fold, X_va_fold = X.iloc[train_idx], X.iloc[val_idx]
        y_tr_fold, y_va_fold = y.iloc[train_idx], y.iloc[val_idx]

        # Apply GPU-accelerated feature engineering
        # create_features returns: res_train, y_train, res_val, y_val, res_test
        f_train, f_y_train, f_val, f_y_val, f_test = create_features(
            X_tr_fold, y_tr_fold, X_va_fold, y_va_fold, X_test
        )

        # Train intent classifier and generate predictions
        # val_pred and te_pred are predicted class labels (e.g., 'DATE', 'CARDINAL', etc.)
        val_pred, te_pred = train_fn(f_train, f_y_train, f_val, f_y_val, f_test)

        # Store predictions for ensembling
        oof_class_preds[val_idx] = val_pred
        test_preds_by_fold.append(te_pred)

        fold_idx += 1

    # 4. Ensemble predictions and normalize strings
    # The ensemble function uses majority voting for classes and applies high-precision rules.
    all_oof_preds = {model_name: oof_class_preds}
    all_test_preds = {model_name: test_preds_by_fold}

    # final_normalized_strings contains the 'after' column values for X_test
    final_normalized_strings = ensemble(
        all_oof_preds=all_oof_preds,
        all_test_preds=all_test_preds,
        y_true_full=y
    )

    # 5. Evaluation and Artifact Generation
    # Calculate intent classification accuracy on the OOF set
    class_accuracy = accuracy_score(y['class'].astype(str), oof_class_preds.astype(str))

    # Ensure the output directory exists
    os.makedirs(OUTPUT_DATA_PATH, exist_ok=True)

    # Save the final submission CSV
    submission_path = os.path.join(OUTPUT_DATA_PATH, "submission.csv")
    submission_df = pd.DataFrame({
        'id': test_ids.values,
        'after': final_normalized_strings
    })
    submission_df.to_csv(submission_path, index=False)

    # Save OOF predictions as an artifact for further analysis
    oof_path = os.path.join(OUTPUT_DATA_PATH, "oof_class_predictions.npy")
    np.save(oof_path, oof_class_preds)

    # 6. Return output information
    output_info = {
        "submission_file_path": submission_path,
        "oof_predictions_path": oof_path,
        "model_scores": {
            f"{model_name}_oof_accuracy": float(class_accuracy)
        }
    }

    return output_info
