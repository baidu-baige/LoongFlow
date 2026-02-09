import os
import pandas as pd
import numpy as np
from typing import Dict
from sklearn.metrics import cohen_kappa_score
from scipy.optimize import minimize

# Import all component functions
from load_data import load_data
from preprocess import preprocess
from get_splitter import get_splitter
from train_and_predict import PREDICTION_ENGINES
from ensemble import ensemble

BASE_DATA_PATH = "/root/workspace/evolux_ml/output/mlebench/aptos2019-blindness-detection/prepared/public"
OUTPUT_DATA_PATH = "output/16326c74-72ad-4b59-ad28-cc76a3d9d373/5/executor/output"

def workflow() -> dict:
    """
    Orchestrates the complete end-to-end machine learning pipeline in production mode.
    Executes a 5-fold stratified cross-validation strategy using EfficientNet-B5 for regression,
    performs 8-view TTA, optimizes thresholds for Quadratic Weighted Kappa, and generates 
    the final submission and metrics.
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(OUTPUT_DATA_PATH):
        os.makedirs(OUTPUT_DATA_PATH, exist_ok=True)

    # 1. Load full dataset
    print("Loading full dataset for production pipeline...")
    X_train, y_train, X_test, test_ids = load_data(validation_mode=False)
    
    # 2. Set up data splitting strategy
    print("Initializing stratified k-fold splitter...")
    splitter = get_splitter(X_train, y_train)
    
    # 3. K-Fold Training and Inference
    # Initialize containers for out-of-fold (OOF) and test predictions
    # oof_predictions will store the continuous regression scores
    oof_predictions = np.zeros(len(X_train), dtype=np.float32)
    test_predictions_list = []
    
    model_name = "efficientnet_b5_tta_regression"
    train_fn = PREDICTION_ENGINES[model_name]
    
    for fold, (train_idx, val_idx) in enumerate(splitter.split(X_train, y_train)):
        print(f"\n--- Starting Fold {fold + 1} / 5 ---")
        
        # Split raw data for this fold
        fold_X_train = X_train.iloc[train_idx]
        fold_y_train = y_train.iloc[train_idx]
        fold_X_val = X_train.iloc[val_idx]
        fold_y_val = y_train.iloc[val_idx]
        
        # a. Preprocess data (converts paths to normalized NumPy arrays: N x 456 x 456 x 3)
        # This includes contour cropping and Ben Graham's enhancement
        print(f"Preprocessing images for fold {fold + 1}...")
        (
            pre_X_train,
            pre_y_train,
            pre_X_val,
            pre_y_val,
            pre_X_test
        ) = preprocess(
            fold_X_train,
            fold_y_train,
            fold_X_val,
            fold_y_val,
            X_test
        )
        
        # b. Train model and predict continuous regression scores with 8-view TTA
        print(f"Training EfficientNet-B5 and performing TTA for fold {fold + 1}...")
        fold_val_out, fold_test_out = train_fn(
            pre_X_train,
            pre_y_train,
            pre_X_val,
            pre_y_val,
            pre_X_test
        )
        
        # Store results for ensembling and CV calculation
        oof_predictions[val_idx] = fold_val_out
        test_predictions_list.append(fold_test_out)

    # 4. Ensemble stage: Optimizes thresholds on full OOF and applies to averaged test scores
    print("\nEnsembling predictions and optimizing thresholds...")
    # Wrap in dictionaries as required by the ensemble function interface
    all_val_outputs = {"kfold_oof": oof_predictions}
    all_test_outputs = {f"fold_{i}": p for i, p in enumerate(test_predictions_list)}
    
    final_test_preds = ensemble(
        all_val_outputs=all_val_outputs,
        all_test_outputs=all_test_outputs,
        y_val=y_train.values
    )

    # 5. Compute Final CV Score (QWK) for the full integrated OOF
    # We perform threshold optimization on the aggregated OOF to report the pipeline CV metric
    def kappa_loss(thresholds, y_true, y_pred):
        t = sorted(thresholds)
        y_digitized = np.clip(np.digitize(y_pred, t), 0, 4)
        return -cohen_kappa_score(y_true, y_digitized, weights='quadratic')

    res = minimize(
        kappa_loss,
        [0.5, 1.5, 2.5, 3.5],
        args=(y_train.values.astype(int), oof_predictions),
        method='Nelder-Mead'
    )
    cv_score = -res.fun
    print(f"Final Pipeline CV Quadratic Weighted Kappa: {cv_score:.4f}")

    # 6. Generate deliverables
    # Save submission file in the format specified: id_code,diagnosis
    submission_df = pd.DataFrame({
        "id_code": test_ids,
        "diagnosis": final_test_preds.astype(int)
    })
    submission_path = os.path.join(OUTPUT_DATA_PATH, "submission.csv")
    submission_df.to_csv(submission_path, index=False)
    
    # Calculate prediction statistics for JSON return
    # oof stats are calculated on continuous scores; test stats on discrete labels
    prediction_stats = {
        "oof": {
            "mean": float(np.mean(oof_predictions)),
            "std": float(np.std(oof_predictions)),
            "min": float(np.min(oof_predictions)),
            "max": float(np.max(oof_predictions)),
        },
        "test": {
            "mean": float(np.mean(final_test_preds)),
            "std": float(np.std(final_test_preds)),
            "min": float(np.min(final_test_preds)),
            "max": float(np.max(final_test_preds)),
        }
    }

    output_info = {
        "submission_file_path": submission_path,
        "model_scores": {f"{model_name}_cv": float(cv_score)},
        "prediction_stats": prediction_stats,
    }
    
    print(f"\nWorkflow complete. Submission saved to: {submission_path}")
    return output_info