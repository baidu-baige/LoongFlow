# -*- coding: utf-8 -*-
"""
LLM Generated Code
"""

import gc
import os

import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata, spearmanr

from create_features import create_features
from cross_validation import cross_validation
from ensemble import ensemble
from load_data import load_data
from train_and_predict import PREDICTION_ENGINES

BASE_DATA_PATH = "/root/workspace/evolux_ml/output/mlebench/google-quest-challenge/prepared/public"
OUTPUT_DATA_PATH = "output/ea4f9f02-c6be-4ad0-a6fd-71930e8fb81e/15/executor/output"


def workflow() -> dict:
    """
    Orchestrates the complete end-to-end machine learning pipeline in production mode.
    
    This function integrates data loading, feature engineering, cross-validation, 
    model training, and rank-based ensembling to generate final deliverables.
    """
    # 1. Load full dataset
    # validation_mode=False is essential for production execution.
    X, y, X_test, test_ids = load_data(validation_mode=False)

    # 2. Set up cross-validation strategy
    # The QuestionGroupKFold ensures that all answers for a single question 
    # are kept together to prevent data leakage.
    splitter = cross_validation(X, y)

    # Initialize storage for predictions across different model architectures
    model_names = list(PREDICTION_ENGINES.keys())
    all_oof_preds = {name: np.zeros(y.shape) for name in model_names}
    all_test_preds = {name: [] for name in model_names}

    # 3. Cross-Validation Loop
    for fold, (train_idx, val_idx) in enumerate(splitter.split(X, y)):
        # a. Split train/validation data for the current fold
        X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
        y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]

        # b. Feature Engineering
        # create_features transforms raw text into token IDs/masks and adds structural features.
        X_tr_t, y_tr_t, X_val_t, y_val_t, X_te_t = create_features(
            X_train_fold, y_train_fold, X_val_fold, y_val_fold, X_test
        )

        # c. Model Training and Prediction
        for name in model_names:
            train_fn = PREDICTION_ENGINES[name]

            # Train the model and obtain Out-of-Fold (OOF) and Test predictions
            val_p, test_p = train_fn(X_tr_t, y_tr_t, X_val_t, y_val_t, X_te_t)

            # Store predictions for ensembling later
            all_oof_preds[name][val_idx] = val_p
            all_test_preds[name].append(test_p)

            # Resource Management: Mandatory cleanup of GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

        # Cleanup fold-specific transformed features to save RAM
        del X_tr_t, y_tr_t, X_val_t, y_val_t, X_te_t
        gc.collect()

    # 4. Ensemble Test Predictions
    # Uses the provided ensemble function (Multi-stage Rank Averaging).
    final_test_preds_df = ensemble(all_oof_preds, all_test_preds, y)

    # 5. Ensemble Out-of-Fold (OOF) Predictions
    # Replicate rank-averaging logic on OOF predictions for robust CV evaluation.
    n_samples, n_targets = y.shape
    ensembled_oof = np.zeros((n_samples, n_targets))

    for col_idx in range(n_targets):
        combined_ranks = np.zeros(n_samples)
        for name in model_names:
            # Using rankdata to handle Spearman-style ranking consistently with the ensemble component
            combined_ranks += rankdata(all_oof_preds[name][:, col_idx])

        rank_min = combined_ranks.min()
        rank_max = combined_ranks.max()

        if rank_max > rank_min:
            ensembled_oof[:, col_idx] = (combined_ranks - rank_min) / (rank_max - rank_min)
        else:
            # Fallback for constant predictions
            ensembled_oof[:, col_idx] = np.mean([all_oof_preds[n][:, col_idx] for n in model_names], axis=0)

    # 6. Evaluation Metrics
    def compute_spearman_scores(y_true, y_pred, columns):
        """Calculates column-wise Spearman correlation and returns per-column and mean scores."""
        per_col = {}
        y_true_arr = y_true.values if hasattr(y_true, 'values') else y_true
        for i, col in enumerate(columns):
            score, _ = spearmanr(y_true_arr[:, i], y_pred[:, i])
            # Handle potential NaNs in Spearman calculation by defaulting to 0
            per_col[col] = float(score) if not np.isnan(score) else 0.0
        overall = float(np.mean(list(per_col.values())))
        return per_col, overall

    model_cv_summary = {}
    for name in model_names:
        _, m_score = compute_spearman_scores(y, all_oof_preds[name], y.columns)
        model_cv_summary[name] = m_score

    per_column_scores, ensemble_score = compute_spearman_scores(y, ensembled_oof, y.columns)
    model_cv_summary['ensemble'] = ensemble_score

    # 7. Deliverables and Artifact Saving
    if not os.path.exists(OUTPUT_DATA_PATH):
        os.makedirs(OUTPUT_DATA_PATH, exist_ok=True)

    # a. Final Submission CSV
    sub_path = os.path.join(OUTPUT_DATA_PATH, 'submission.csv')
    submission = final_test_preds_df.copy()
    submission.insert(0, 'qa_id', test_ids.values)
    submission.to_csv(sub_path, index=False)

    # b. OOF Predictions Artifact
    oof_path = os.path.join(OUTPUT_DATA_PATH, 'oof_predictions.csv')
    oof_df = pd.DataFrame(ensembled_oof, columns=y.columns)
    # Ensure qa_id alignment for OOF (X contains qa_id from original load_data)
    oof_df.insert(0, 'qa_id', X['qa_id'].values)
    oof_df.to_csv(oof_path, index=False)

    # Return final results in the required JSON-serializable format
    output_info = {
        "submission_file_path": sub_path,
        "oof_file_path": oof_path,
        "model_scores": model_cv_summary,
        "per_column_scores": per_column_scores
    }

    return output_info
