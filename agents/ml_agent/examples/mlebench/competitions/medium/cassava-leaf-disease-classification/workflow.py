import os
import gc
import numpy as np
import pandas as pd
import torch

# Import all component functions
from load_data import load_data
from preprocess import preprocess
from get_splitter import get_splitter
from train_and_predict import PREDICTION_ENGINES
from ensemble import ensemble

BASE_DATA_PATH = "/mnt/pfs/loongflow/devmachine/h20-12/evolux/output/mlebench/cassava-leaf-disease-classification/prepared/public"
OUTPUT_DATA_PATH = "output/502e395f-5bca-4b30-9a81-fc7b49a1e544/3/executor/output"

def workflow() -> dict:
    """
    Orchestrates the complete end-to-end machine learning pipeline in production mode.
    Executes a 5-fold cross-validation loop training an ensemble of ConvNeXt and Swin models.
    """
    # Create output directory for artifacts
    os.makedirs(OUTPUT_DATA_PATH, exist_ok=True)

    # 1. Load complete dataset (Production mode)
    print("Stage 1: Loading full dataset...")
    X_train_all, y_train_all, X_test_meta, test_ids = load_data(validation_mode=False)

    # 2. Set up data splitting strategy (Stratified 5-Fold)
    print("Stage 2: Setting up 5-fold cross-validation...")
    splitter = get_splitter(X_train_all, y_train_all)
    
    num_train = len(X_train_all)
    num_test = len(X_test_meta)
    num_classes = 5
    
    # Initialize separate buffers (numpy arrays) for each model architecture as per technical spec
    # This ensures decoupled buffer management for the weighted ensembler
    oof_buffers = {
        "convnext": np.zeros((num_train, num_classes), dtype=np.float32),
        "swin": np.zeros((num_train, num_classes), dtype=np.float32)
    }
    test_buffers = {
        "convnext": np.zeros((num_test, num_classes), dtype=np.float32),
        "swin": np.zeros((num_test, num_classes), dtype=np.float32)
    }

    # 3. Sequential 5-Fold Cross-Validation Loop
    print("Stage 3: Executing 5-fold CV loop...")
    for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(X_train_all, y_train_all)):
        print(f"\n--- Processing Fold {fold_idx + 1}/5 ---")
        
        # Define fold subsets
        X_train_fold = X_train_all.iloc[train_idx]
        y_train_fold = y_train_all.iloc[train_idx]
        X_val_fold = X_train_all.iloc[val_idx]
        y_val_fold = y_train_all.iloc[val_idx]

        # a. Preprocess images into model-ready high-resolution tensors
        print(f"Fold {fold_idx + 1}: Preprocessing image data...")
        (
            X_train_tensor,
            y_train_tensor,
            X_val_tensor,
            y_val_tensor,
            X_test_tensor
        ) = preprocess(X_train_fold, y_train_fold, X_val_fold, y_val_fold, X_test_meta)

        # b. Train High-Res Vision Ensemble (ConvNeXt-Base & Swin-Base)
        # This engine executes DDP internally and returns decoupled model predictions
        print(f"Fold {fold_idx + 1}: Training models and generating predictions...")
        val_preds_dict, test_preds_dict = PREDICTION_ENGINES["high_res_vision_ensemble"](
            X_train_tensor,
            y_train_tensor,
            X_val_tensor,
            y_val_tensor,
            X_test_tensor
        )

        # c. Record OOF predictions and accumulate test predictions for each model
        for model_key in ["convnext", "swin"]:
            oof_buffers[model_key][val_idx] = val_preds_dict[model_key]
            # Accumulate test predictions (simple averaging across folds)
            test_buffers[model_key] += test_preds_dict[model_key] / 5.0

        # d. Resource Management: Clear GPU memory and garbage collect after each model/fold
        print(f"Fold {fold_idx + 1}: Cleaning up resources...")
        del X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor, X_test_tensor
        del X_train_fold, y_train_fold, X_val_fold, y_val_fold
        del val_preds_dict, test_preds_dict
        torch.cuda.empty_cache()
        gc.collect()

    # 4. Final Ensemble and Weight Optimization
    # The ensemble function is provided with the full decoupled OOF dictionary for Nelder-Mead optimization
    print("\nStage 4: Performing weighted ensemble optimization...")
    y_true_tensor = torch.tensor(y_train_all.values, dtype=torch.long)
    final_test_labels = ensemble(oof_buffers, test_buffers, y_true_tensor)

    # 5. Generate deliverables
    print("Stage 5: Generating submission.csv and computing statistics...")
    submission_df = pd.DataFrame({
        "image_id": test_ids,
        "label": final_test_labels
    })
    submission_path = os.path.join(OUTPUT_DATA_PATH, "submission.csv")
    submission_df.to_csv(submission_path, index=False)
    
    # Compute OOF labels for summary statistics (using simple average/sum for representation)
    combined_oof_probs = oof_buffers["convnext"] + oof_buffers["swin"]
    oof_labels = np.argmax(combined_oof_probs, axis=1)

    output_info = {
        "submission_file_path": submission_path,
        "prediction_stats": {
            "oof": {
                "mean": float(np.mean(oof_labels)),
                "std": float(np.std(oof_labels)),
                "min": float(np.min(oof_labels)),
                "max": float(np.max(oof_labels))
            },
            "test": {
                "mean": float(np.mean(final_test_labels)),
                "std": float(np.std(final_test_labels)),
                "min": float(np.min(final_test_labels)),
                "max": float(np.max(final_test_labels))
            },
        },
    }

    # Final resource cleanup
    del oof_buffers, test_buffers, submission_df, combined_oof_probs
    torch.cuda.empty_cache()
    gc.collect()

    print("Pipeline execution completed successfully.")
    return output_info