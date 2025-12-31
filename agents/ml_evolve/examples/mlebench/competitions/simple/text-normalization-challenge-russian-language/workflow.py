# -*- coding: utf-8 -*-
"""
LLM Generated Code
"""

import os

import numpy as np
import pandas as pd

from create_features import create_features
from cross_validation import cross_validation
from ensemble import ensemble
# Assume all component functions are available for import
from load_data import load_data
from train_and_predict import PREDICTION_ENGINES

BASE_DATA_PATH = "/root/workspace/evolux/output/mlebench/text-normalization-challenge-russian-language/prepared/public"
OUTPUT_DATA_PATH = "output/dce1a922-fb4b-4006-9d03-6f53b7ea0718/1/executor/output"


def workflow() -> dict:
    """
    Executes the complete machine learning workflow to generate the required deliverables.
    This function orchestrates the entire pipeline, from data processing to model
    training and evaluation. Its primary purpose is to produce and consolidate all
    the final artifacts specified in the task description.

    Returns:
        dict: A dictionary containing all the deliverables required by the task 
              description. It serves as a structured, serializable manifest of the results.
    """
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DATA_PATH, exist_ok=True)

    # Step 1: Load all data
    print("=" * 60)
    print("STEP 1: Loading data...")
    print("=" * 60)
    X, y, X_test, test_ids = load_data()

    # Ensure all text columns are string type to avoid conversion issues
    print("\nConverting text columns to string type...")
    if 'before' in X.columns:
        X['before'] = X['before'].astype(str)
    if 'before' in X_test.columns:
        X_test['before'] = X_test['before'].astype(str)

    # Convert y to string
    if isinstance(y, pd.Series):
        y = y.astype(str)

    # Ensure sentence_id and token_id are proper integers (handle any special characters)
    for df in [X, X_test]:
        if 'sentence_id' in df.columns:
            df['sentence_id'] = pd.to_numeric(df['sentence_id'], errors='coerce').fillna(0).astype(int)
        if 'token_id' in df.columns:
            df['token_id'] = pd.to_numeric(df['token_id'], errors='coerce').fillna(0).astype(int)

    print(f"\nData loaded successfully:")
    print(f"  Training samples: {len(X):,}")
    print(f"  Test samples: {len(X_test):,}")

    # Step 2: Set up cross-validation strategy
    print("\n" + "=" * 60)
    print("STEP 2: Setting up cross-validation...")
    print("=" * 60)
    cv = cross_validation(X, y)
    n_splits = cv.get_n_splits()
    print(f"Cross-validation configured with {n_splits} folds")

    # Step 3: Initialize storage for predictions
    print("\n" + "=" * 60)
    print("STEP 3: Initializing prediction storage...")
    print("=" * 60)

    # Get list of models from PREDICTION_ENGINES
    model_names = list(PREDICTION_ENGINES.keys())
    print(f"Models to train: {model_names}")

    # Initialize OOF predictions storage (dict: model_name -> array)
    all_oof_preds = {model_name: np.empty(len(X), dtype=object) for model_name in model_names}

    # Initialize test predictions storage (dict: model_name -> list of fold predictions)
    all_test_preds = {model_name: [] for model_name in model_names}

    # Track model scores
    model_fold_scores = {model_name: [] for model_name in model_names}

    # Step 4: Cross-validation loop
    print("\n" + "=" * 60)
    print("STEP 4: Running cross-validation...")
    print("=" * 60)

    # Get groups for splitting (sentence_id)
    groups = X['sentence_id'].values if 'sentence_id' in X.columns else None

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y, groups)):
        print(f"\n{'=' * 40}")
        print(f"FOLD {fold_idx + 1}/{n_splits}")
        print(f"{'=' * 40}")
        print(f"  Train size: {len(train_idx):,}")
        print(f"  Validation size: {len(val_idx):,}")

        # Split data for this fold
        X_train_fold = X.iloc[train_idx].copy().reset_index(drop=True)
        y_train_fold = y.iloc[train_idx].copy().reset_index(drop=True)
        X_val_fold = X.iloc[val_idx].copy().reset_index(drop=True)
        y_val_fold = y.iloc[val_idx].copy().reset_index(drop=True)

        # Step 4b: Create features for this fold
        print(f"\n  Creating features for fold {fold_idx + 1}...")
        try:
            X_train_transformed, y_train_transformed, X_val_transformed = create_features(
                X_train_fold, y_train_fold, X_val_fold
            )
        except Exception as e:
            print(f"  Warning: Feature creation failed with error: {e}")
            print(f"  Using original features...")
            X_train_transformed = X_train_fold.copy()
            y_train_transformed = y_train_fold.copy()
            X_val_transformed = X_val_fold.copy()

        # Also transform test data using training data from this fold
        print(f"  Creating features for test data...")
        try:
            _, _, X_test_transformed = create_features(
                X_train_fold, y_train_fold, X_test.copy()
            )
        except Exception as e:
            print(f"  Warning: Test feature creation failed with error: {e}")
            print(f"  Using original test features...")
            X_test_transformed = X_test.copy()

        # Step 4c: Train each model and get predictions
        for model_name in model_names:
            print(f"\n  Training model: {model_name}")

            # Get the prediction function
            predict_fn = PREDICTION_ENGINES[model_name]

            try:
                # Train and predict
                val_preds, test_preds = predict_fn(
                    X_train_transformed,
                    y_train_transformed,
                    X_val_transformed,
                    y_val_fold,
                    X_test_transformed
                )

                # Ensure predictions are string type
                val_preds = np.array(
                    [str(p) if p is not None and not (isinstance(p, float) and pd.isna(p)) else '' for p in val_preds])
                test_preds = np.array(
                    [str(p) if p is not None and not (isinstance(p, float) and pd.isna(p)) else '' for p in test_preds])

                # Store OOF predictions
                all_oof_preds[model_name][val_idx] = val_preds

                # Store test predictions for this fold
                all_test_preds[model_name].append(test_preds)

                # Calculate fold accuracy
                y_val_arr = y_val_fold.values if isinstance(y_val_fold, pd.Series) else np.array(y_val_fold)
                y_val_arr = np.array([str(v) for v in y_val_arr])
                fold_accuracy = np.mean(val_preds == y_val_arr)
                model_fold_scores[model_name].append(fold_accuracy)
                print(f"    Fold {fold_idx + 1} accuracy: {fold_accuracy:.4f}")

            except Exception as e:
                print(f"    Error training {model_name}: {e}")
                # Fill with empty predictions on error
                all_oof_preds[model_name][val_idx] = np.array(['' for _ in range(len(val_idx))])
                all_test_preds[model_name].append(np.array(['' for _ in range(len(X_test))]))
                model_fold_scores[model_name].append(0.0)

    # Step 5: Calculate final CV scores
    print("\n" + "=" * 60)
    print("STEP 5: Calculating final CV scores...")
    print("=" * 60)

    model_cv_scores = {}
    for model_name in model_names:
        fold_scores = model_fold_scores[model_name]
        mean_score = np.mean(fold_scores)
        std_score = np.std(fold_scores)
        model_cv_scores[model_name] = {
            'mean': float(mean_score),
            'std': float(std_score),
            'fold_scores': [float(s) for s in fold_scores]
        }
        print(f"  {model_name}: {mean_score:.4f} (+/- {std_score:.4f})")

    # Step 6: Ensemble predictions
    print("\n" + "=" * 60)
    print("STEP 6: Ensembling predictions...")
    print("=" * 60)

    try:
        final_test_predictions = ensemble(
            all_oof_preds,
            all_test_preds,
            y
        )
    except Exception as e:
        print(f"Ensemble failed with error: {e}")
        print("Using first model's predictions as fallback...")
        # Fallback: use majority vote from first model's fold predictions
        first_model = model_names[0]
        fold_preds = all_test_preds[first_model]
        if fold_preds:
            # Simple average across folds (for text, use first fold)
            final_test_predictions = fold_preds[0]
        else:
            final_test_predictions = np.array(['' for _ in range(len(X_test))])

    # Step 7: Create submission file
    print("\n" + "=" * 60)
    print("STEP 7: Creating submission file...")
    print("=" * 60)

    # Create submission DataFrame
    test_ids_values = test_ids.values if isinstance(test_ids, pd.Series) else test_ids

    submission = pd.DataFrame({
        'id': test_ids_values,
        'after': final_test_predictions
    })

    # Ensure no NaN values in 'after' column
    submission['after'] = submission['after'].fillna('')

    # Convert any non-string values to string
    submission['after'] = submission['after'].astype(str)

    # Replace 'nan' strings with empty string
    submission['after'] = submission['after'].replace('nan', '')

    # Verify submission format
    print(f"\nSubmission shape: {submission.shape}")
    print(f"Expected rows: ~1,059,191")
    print(f"Actual rows: {len(submission):,}")

    # Check for NaN values
    nan_count = submission['after'].isna().sum()
    empty_count = (submission['after'] == '').sum()
    print(f"NaN values in 'after': {nan_count}")
    print(f"Empty strings in 'after': {empty_count}")

    # Check id format (should be sentence_id_token_id)
    sample_ids = submission['id'].head(5).tolist()
    print(f"Sample IDs: {sample_ids}")

    # Sample predictions
    sample_preds = submission['after'].head(5).tolist()
    print(f"Sample predictions: {sample_preds}")

    # Save submission file with proper quoting
    submission_path = os.path.join(OUTPUT_DATA_PATH, "submission.csv")
    submission.to_csv(submission_path, index=False, quoting=1)  # quoting=1 is QUOTE_ALL
    print(f"\nSubmission saved to: {submission_path}")

    # Verify the saved file
    try:
        saved_submission = pd.read_csv(submission_path)
        print(f"Verified saved submission: {len(saved_submission):,} rows")
    except Exception as e:
        print(f"Warning: Could not verify saved submission: {e}")

    # Step 8: Calculate overall OOF accuracy
    print("\n" + "=" * 60)
    print("STEP 8: Final evaluation...")
    print("=" * 60)

    overall_oof_scores = {}
    for model_name in model_names:
        try:
            oof_preds = all_oof_preds[model_name]
            y_arr = y.values if isinstance(y, pd.Series) else np.array(y)

            # Convert to string for comparison
            oof_preds_str = np.array([str(p) if p is not None and str(p) != 'nan' else '' for p in oof_preds])
            y_arr_str = np.array([str(t) if t is not None and str(t) != 'nan' else '' for t in y_arr])

            overall_accuracy = np.mean(oof_preds_str == y_arr_str)
            overall_oof_scores[model_name] = float(overall_accuracy)
            print(f"  {model_name} overall OOF accuracy: {overall_accuracy:.4f}")
        except Exception as e:
            print(f"  Error calculating OOF score for {model_name}: {e}")
            overall_oof_scores[model_name] = 0.0

    # Create output info dictionary
    output_info = {
        "submission_file_path": submission_path,
        "model_cv_scores": model_cv_scores,
        "overall_oof_scores": overall_oof_scores,
        "n_train_samples": int(len(X)),
        "n_test_samples": int(len(X_test)),
        "n_folds": n_splits,
        "models_used": model_names,
        "submission_rows": int(len(submission))
    }

    print("\n" + "=" * 60)
    print("WORKFLOW COMPLETE")
    print("=" * 60)
    print(f"\nOutput info: {output_info}")

    return output_info
