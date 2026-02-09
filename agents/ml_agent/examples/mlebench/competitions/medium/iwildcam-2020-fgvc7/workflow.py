import os
import pandas as pd
import numpy as np
from load_data import load_data
from preprocess import preprocess
from get_splitter import get_splitter
from train_and_predict import PREDICTION_ENGINES
from ensemble import ensemble

# Standard technical paths
BASE_DATA_PATH = "/mnt/pfs/loongflow/devmachine/h20-09/evolux/output/mlebench/iwildcam-2020-fgvc7/prepared/public"
OUTPUT_DATA_PATH = "output/9508c267-92c9-4fd0-91c8-90efc0fba263/1/executor/output"

def workflow() -> dict:
    """
    Orchestrates the complete end-to-end machine learning pipeline in production mode.
    """
    print("Starting production workflow...")

    # 1. Load full dataset
    # validation_mode=False loads the complete datasets (approx 217k train, 63k test)
    X_train_full, y_train_full, X_test_full, test_ids = load_data(validation_mode=False)
    
    # 2. Set up data splitting strategy
    # Uses GroupKFold based on 'location' to ensure generalization to unseen sites
    splitter = get_splitter(X_train_full, y_train_full)
    num_folds = splitter.get_n_splits(X_train_full, y_train_full)
    
    # Storage for ensembling and OOF calculation
    # num_classes is 676 as defined in train_and_predict component
    num_classes = 676
    all_test_probs = {}
    oof_probs = np.zeros((len(X_train_full), num_classes))
    
    # 3. Execute Cross-Validation Pipeline
    for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(X_train_full, y_train_full)):
        print(f"--- Processing Fold {fold_idx + 1}/{num_folds} ---")
        
        # Split features and targets
        X_train, y_train = X_train_full.iloc[train_idx], y_train_full.iloc[train_idx]
        X_val, y_val = X_train_full.iloc[val_idx], y_train_full.iloc[val_idx]
        
        # a. Preprocess
        # Generates PyTorch WildCamDatasets with MegaDetector cropping and augmentations
        X_train_p, y_train_p, X_val_p, y_val_p, X_test_p = preprocess(
            X_train, y_train, X_val, y_val, X_test_full
        )
        
        # b. Train and Predict
        # Uses ConvNeXt-Base with DistributedDataParallel on available GPUs
        model_fn = PREDICTION_ENGINES["convnext_base"]
        val_probs, test_probs = model_fn(X_train_p, y_train_p, X_val_p, y_val_p, X_test_p)
        
        # Store probabilities for ensembling
        all_test_probs[f"fold_{fold_idx}"] = test_probs
        oof_probs[val_idx] = val_probs
        
        print(f"Fold {fold_idx + 1} completed.")

    # 4. Ensemble Predictions
    # Combines fold predictions via soft-voting and applies sequence-level temporal smoothing.
    # The ensemble function internally re-loads metadata for sequence IDs.
    print("Ensembling fold predictions and applying sequence-level smoothing...")
    final_test_classes = ensemble(
        all_val_preds={}, # Provided ensemble function focuses on test set aggregation
        all_test_preds=all_test_probs,
        y_val=y_train_full
    )
    
    # 5. Compute Prediction Statistics
    # Calculate categorical statistics for both OOF and Test sets
    oof_classes = np.argmax(oof_probs, axis=1)
    
    prediction_stats = {
        "oof": {
            "mean": float(np.mean(oof_classes)),
            "std": float(np.std(oof_classes)),
            "min": float(np.min(oof_classes)),
            "max": float(np.max(oof_classes)),
        },
        "test": {
            "mean": float(np.mean(final_test_classes)),
            "std": float(np.std(final_test_classes)),
            "min": float(np.min(final_test_classes)),
            "max": float(np.max(final_test_classes)),
        }
    }
    
    # 6. Generate Deliverables
    # Create submission file with strict 'Id,Category' formatting
    submission_df = pd.DataFrame({
        "Id": test_ids.values,
        "Category": final_test_classes.astype(int)
    })
    
    submission_file_path = os.path.join(OUTPUT_DATA_PATH, "submission.csv")
    submission_df.to_csv(submission_file_path, index=False)
    
    # Save OOF predictions for downstream analysis if needed
    np.save(os.path.join(OUTPUT_DATA_PATH, "oof_probs.npy"), oof_probs)
    
    print(f"Workflow complete. Submission saved to {submission_file_path}")
    
    return {
        "submission_file_path": submission_file_path,
        "prediction_stats": prediction_stats,
        "num_folds_processed": num_folds
    }