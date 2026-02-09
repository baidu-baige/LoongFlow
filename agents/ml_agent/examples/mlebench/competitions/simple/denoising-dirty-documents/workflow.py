import os
import numpy as np
import csv
from typing import Dict, List
from load_data import load_data
from preprocess import preprocess
from get_splitter import get_splitter
from train_and_predict import PREDICTION_ENGINES
from ensemble import ensemble

BASE_DATA_PATH = "/mnt/pfs/loongflow/devmachine/h20-04/evolux/output/mlebench/denoising-dirty-documents/prepared/public"
OUTPUT_DATA_PATH = "output/72c59496-c374-4998-9558-a1c06c176e1b/1/executor/output"


def workflow() -> dict:
    """
    Orchestrates the complete end-to-end machine learning pipeline in production mode.
    Executes a 5-fold cross-validation strategy, ensembles predictions, and generates
    the final competition submission.
    """
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DATA_PATH, exist_ok=True)

    # 1. Load full dataset
    print("Stage 1: Loading full dataset...")
    X_train_raw, y_train_raw, X_test_raw, test_ids = load_data(validation_mode=False)

    # 2. Set up data splitting strategy
    print("Stage 2: Setting up 5-fold cross-validation...")
    splitter = get_splitter(X_train_raw, y_train_raw)

    all_test_preds_dict = {}
    oof_preds_flat_list = []

    # 3. Cross-validation loop
    for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(X_train_raw)):
        print(f"\n--- Starting Fold {fold_idx + 1}/5 ---")

        # Split raw lists according to fold indices
        X_tr = [X_train_raw[i] for i in train_idx]
        y_tr = [y_train_raw[i] for i in train_idx]
        X_va = [X_train_raw[i] for i in val_idx]
        y_va = [y_train_raw[i] for i in val_idx]

        # Preprocess: Extract patches for training, pad validation/test for inference
        print(f"Fold {fold_idx + 1}: Preprocessing images...")
        X_tr_p, y_tr_p, X_va_p, y_va_p, X_te_p = preprocess(
            X_tr, y_tr, X_va, y_va, X_test_raw
        )

        # Train and Predict: Using the primary denoising engine
        print(f"Fold {fold_idx + 1}: Training model and generating predictions...")
        model_fn = PREDICTION_ENGINES["unet_denoiser"]
        val_preds_p, test_preds_p = model_fn(X_tr_p, y_tr_p, X_va_p, y_va_p, X_te_p)

        # Store test predictions for later ensembling
        all_test_preds_dict[f"fold_{fold_idx}"] = test_preds_p

        # Collect Out-of-Fold (OOF) predictions for evaluation
        # We crop the padded validation predictions back to their original spatial dimensions
        for i in range(len(val_idx)):
            h_orig, w_orig = X_va[i].shape
            # val_preds_p shape: (N_val, 1, H_pad, W_pad)
            pred_crop = val_preds_p[i, 0, :h_orig, :w_orig]
            oof_preds_flat_list.append(pred_crop.flatten())

    # 4. Ensemble predictions from all folds
    # Simple arithmetic mean across the 5 models/folds
    print("\nStage 4: Ensembling test set predictions...")
    final_test_preds = ensemble(
        all_val_preds={},
        all_test_preds=all_test_preds_dict,
        y_val=np.zeros((0, 1, 1, 1))  # Dummy y_val for signature compatibility
    )

    # 5. Compute prediction statistics for deliverables
    print("Stage 5: Computing prediction statistics...")
    oof_all = np.concatenate(oof_preds_flat_list)
    test_all = final_test_preds.flatten()

    prediction_stats = {
        "oof": {
            "mean": float(np.mean(oof_all)),
            "std": float(np.std(oof_all)),
            "min": float(np.min(oof_all)),
            "max": float(np.max(oof_all)),
        },
        "test": {
            "mean": float(np.mean(test_all)),
            "std": float(np.std(test_all)),
            "min": float(np.min(test_all)),
            "max": float(np.max(test_all)),
        }
    }

    # 6. Generate final submission file
    # Format: id, value where id is image_row_col (1-indexed)
    submission_path = os.path.join(OUTPUT_DATA_PATH, "submission.csv")
    print(f"Stage 6: Writing final submission to {submission_path}...")

    with open(submission_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'value'])

        # Iterate through each test image and its corresponding ensembled prediction
        for i, test_id in enumerate(test_ids):
            h_orig, w_orig = X_test_raw[i].shape
            # final_test_preds shape: (N_test, 1, H_pad, W_pad)
            # Crop the prediction back to the original image boundaries
            pred_img = final_test_preds[i, 0, :h_orig, :w_orig]

            # Write each pixel as a row in the submission format
            for r in range(h_orig):
                for c in range(w_orig):
                    # Format: {img_id}_{1-based_row}_{1-based_col}
                    writer.writerow([f"{test_id}_{r + 1}_{c + 1}", pred_img[r, c]])

    print("Workflow complete.")

    return {
        "submission_file_path": submission_path,
        "prediction_stats": prediction_stats
    }