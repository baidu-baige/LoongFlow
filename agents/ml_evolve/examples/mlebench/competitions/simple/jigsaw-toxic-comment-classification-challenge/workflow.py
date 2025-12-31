# -*- coding: utf-8 -*-
"""
LLM Generated Code
"""

import os

import numpy as np
import pandas as pd
import torch

from create_features import create_features
from cross_validation import cross_validation
from ensemble import ensemble
from load_data import load_data
from train_and_predict import PREDICTION_ENGINES

BASE_DATA_PATH = "/root/workspace/evolux/output/mlebench/jigsaw-toxic-comment-classification-challenge/prepared/public"
OUTPUT_DATA_PATH = "output/40d01db3-cd9d-46e2-8d9c-d192fb8addff/1/executor/output"


def workflow() -> dict:
    """
    Orchestrates the complete end-to-end machine learning pipeline for the 
    Jigsaw Toxic Comment Classification Challenge.
    """
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DATA_PATH, exist_ok=True)

    # 1. Load full dataset
    X, y, X_test, test_ids = load_data(validation_mode=False)
    target_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

    # 2. Set up cross-validation strategy
    cv = cross_validation(X, y)

    # Containers for predictions
    # all_oof_preds: {model_name: oof_df}
    # all_test_preds: {model_name: [fold1_test_preds, fold2_test_preds, ...]}
    model_name = "bert_transformer"
    all_test_preds = {model_name: []}

    # Initialize OOF dataframe for the model
    oof_preds = pd.DataFrame(np.zeros((len(X), len(target_cols))), index=X.index, columns=target_cols)

    # 3. Cross-validation loop
    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        # Split data for this fold
        X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
        y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]

        # a. Feature Engineering (Tokenization)
        # Note: create_features uses the whole X_test in every fold which is standard for inference averaging
        X_train_f, y_train_f, X_val_f, y_val_f, X_test_f = create_features(
            X_train_fold, y_train_fold, X_val_fold, y_val_fold, X_test
        )

        # b. Train and Predict
        # Using the bert_transformer engine
        val_preds, test_preds = PREDICTION_ENGINES[model_name](
            X_train_f, y_train_f, X_val_f, y_val_f, X_test_f
        )

        # c. Collect results
        oof_preds.iloc[val_idx] = val_preds.values
        all_test_preds[model_name].append(test_preds)

        # Resource Management: Clear GPU cache between folds
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Create dictionary for ensembling
    all_oof_preds = {model_name: oof_preds}

    # 4. Ensemble predictions
    # This averages across folds for the BERT model (and potentially multiple models if defined)
    final_test_preds = ensemble(all_oof_preds, all_test_preds, y)

    # 5. Generate Submission File
    submission = pd.DataFrame(index=test_ids.index)
    submission['id'] = test_ids.values
    for col in target_cols:
        submission[col] = final_test_preds[col].values

    submission_file_path = os.path.join(OUTPUT_DATA_PATH, "submission.csv")
    submission.to_csv(submission_file_path, index=False)

    # 6. Calculate CV Score (Mean Column-wise ROC AUC)
    from sklearn.metrics import roc_auc_score
    cv_score = roc_auc_score(y.values, oof_preds.values, average='macro')

    # Save artifacts and return metadata
    output_info = {
        "submission_file_path": submission_file_path,
        "model_scores": {model_name: float(cv_score)},
        "n_folds": 5,
        "status": "success"
    }

    return output_info
