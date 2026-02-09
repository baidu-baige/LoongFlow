import os
import numpy as np
import pandas as pd
import torch
from load_data import load_data
from preprocess import preprocess
from get_splitter import get_splitter
from train_and_predict import PREDICTION_ENGINES
from ensemble import ensemble

BASE_DATA_PATH = "/mnt/pfs/loongflow/devmachine/h20-01/evolux/output/mlebench/dogs-vs-cats-redux-kernels-edition/prepared/public"
OUTPUT_DATA_PATH = "output/25e7371d-bfe6-47c9-b200-bdf664ef9932/2/executor/output"

def workflow() -> dict:
    """
    Orchestrates the complete end-to-end machine learning pipeline in production mode.

    This implementation executes a 5-fold cross-validation strategy, utilizing dual GPUs 
    via DDP for high-capacity model training, followed by temperature-scaled ensembling.
    """
    print("Workflow: Starting production pipeline execution.")

    # 1. Load full dataset
    X_train_full, y_train_full, X_test, test_ids = load_data(validation_mode=False)

    # 2. Set up data splitting strategy
    splitter = get_splitter(X_train_full, y_train_full)

    # 3. Initialize prediction containers
    # oof_preds: Stores validation predictions for the entire training set
    # test_preds_list: Stores test predictions from each fold's model ensemble
    oof_preds = np.zeros(len(y_train_full))
    test_preds_list = []

    # 4. Cross-Validation Loop
    print(f"Workflow: Beginning {splitter.get_n_splits()}-fold cross-validation.")
    for fold, (train_idx, val_idx) in enumerate(splitter.split(X_train_full, y_train_full)):
        print(f"\n--- Processing Fold {fold + 1}/{splitter.get_n_splits()} ---")
        
        # Split train/validation data for this fold
        X_tr, y_tr = X_train_full.iloc[train_idx], y_train_full.iloc[train_idx]
        X_va, y_va = X_train_full.iloc[val_idx], y_train_full.iloc[val_idx]
        
        # Apply preprocessing (returns ModelReadyLoaders)
        train_loader, _, val_loader, _, test_loader = preprocess(X_tr, y_tr, X_va, y_va, X_test)
        
        # Train high-capacity ensemble (ConvNeXt, Swin, ViT) using DDP
        # engine returns (fold_val_preds, fold_test_preds)
        engine = PREDICTION_ENGINES["vision_ensemble_ddp"]
        fold_val_preds, fold_test_preds = engine(
            train_loader, y_tr, val_loader, y_va, test_loader
        )
        
        # Store fold results
        oof_preds[val_idx] = fold_val_preds
        test_preds_list.append(fold_test_preds)
        
        # Memory Management: Clear VRAM cache between folds
        torch.cuda.empty_cache()
        print(f"--- Fold {fold + 1} Complete ---")

    # 5. Consolidate and Ensemble
    # Average test predictions across folds to create the base test prediction
    mean_test_preds = np.mean(test_preds_list, axis=0)

    # Use ensemble function for Temperature Scaling calibration and weighted optimization
    # Even with one consolidated CV model, this step ensures optimal calibration
    final_test_preds = ensemble(
        all_val_preds={"cv_ensemble": oof_preds},
        all_test_preds={"cv_ensemble": mean_test_preds},
        y_val=y_train_full
    )

    # 6. Generate Deliverables
    # Ensure output directory exists
    os.makedirs(OUTPUT_DATA_PATH, exist_ok=True)
    
    # Save submission CSV
    submission_df = pd.DataFrame({
        "id": test_ids,
        "label": final_test_preds
    })
    submission_path = os.path.join(OUTPUT_DATA_PATH, "submission.csv")
    submission_df.to_csv(submission_path, index=False)
    print(f"Workflow: Submission file saved to {submission_path}")

    # Compute prediction statistics for the return payload
    # Convert numpy types to native Python floats for JSON serialization
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

    output_info = {
        "submission_file_path": submission_path,
        "prediction_stats": prediction_stats,
    }

    print("Workflow complete. Returning results.")
    return output_info