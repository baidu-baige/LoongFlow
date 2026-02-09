import os
import pandas as pd
import numpy as np
from load_data import load_data
from preprocess import preprocess
from get_splitter import get_splitter
from train_and_predict import PREDICTION_ENGINES
from ensemble import ensemble

BASE_DATA_PATH = "/mnt/pfs/loongflow/devmachine/h20-11/evolux/output/mlebench/rsna-miccai-brain-tumor-radiogenomic-classification/prepared/public"
OUTPUT_DATA_PATH = "output/dafb557f-655e-4395-9835-6f75549a5b27/1/executor/output"

def workflow() -> dict:
    """
    Orchestrates the complete end-to-end machine learning pipeline in production mode.

    This function integrates all pipeline components (data loading, preprocessing, 
    data splitting, model training, and ensembling) to generate final deliverables 
    specified in the task description.
    """
    # 1. Load full dataset with load_data(validation_mode=False)
    print("Workflow: Starting data loading...")
    X_train_all, y_train_all, X_test_all, test_ids = load_data(validation_mode=False)
    
    # 2. Set up data splitting strategy with get_splitter()
    print("Workflow: Initializing cross-validation splitter...")
    splitter = get_splitter(X_train_all, y_train_all)
    
    # Technical Specification requires multi-modality training
    modalities = ['FLAIR', 'T1w', 'T1wCE', 'T2w']
    all_modality_oof = {}
    all_modality_test = {}
    
    # Ensure production output directory exists
    os.makedirs(OUTPUT_DATA_PATH, exist_ok=True)

    # 3. Training loop: Iterate through each modality and each fold
    # As per Technical Specification: Sequential training of 4 modality models across 5 folds.
    for m_idx, m_name in enumerate(modalities):
        print(f"\nWorkflow: Processing MRI Modality -> {m_name}")
        
        # Buffers for Out-of-Fold (OOF) and Test predictions for current modality
        oof_preds = np.zeros(len(y_train_all), dtype=np.float32)
        test_preds_accum = np.zeros(len(test_ids), dtype=np.float32)
        
        # Slice X to pass only the specific modality to the pipeline
        # Preprocess is hardcoded to take the first channel, so we provide the target channel at index 0
        X_train_m = X_train_all[:, [m_idx], :, :, :]
        X_test_m = X_test_all[:, [m_idx], :, :, :]
        
        n_splits = splitter.get_n_splits()
        
        for fold, (train_idx, val_idx) in enumerate(splitter.split(X_train_m, y_train_all)):
            print(f"Workflow: Modality {m_name} | Running Fold {fold+1}/{n_splits}...")
            
            # Split data into training and validation subsets for the current fold
            X_tr, X_val = X_train_m[train_idx], X_train_m[val_idx]
            y_tr, y_val = y_train_all[train_idx], y_train_all[val_idx]
            
            # a. Apply preprocess() to this fold
            # This handles Intensity Standardisation and 3D Augmentations
            X_tr_p, y_tr_p, X_val_p, y_val_p, X_te_p = preprocess(
                X_tr, y_tr, X_val, y_val, X_test_m
            )
            
            # b. Train model and collect predictions
            # Using the 3D-aware EfficientNet-B0 engine
            train_fn = PREDICTION_ENGINES["efficientnet_b0_3d"]
            val_p, test_p = train_fn(X_tr_p, y_tr_p, X_val_p, y_val_p, X_te_p)
            
            # Update OOF predictions for the validation segment
            oof_preds[val_idx] = val_p
            # Accumulate test predictions for averaging after all folds
            test_preds_accum += test_p / n_splits
            
        all_modality_oof[m_name] = oof_preds
        all_modality_test[m_name] = test_preds_accum

    # 4. Ensemble predictions from all models
    # Uses weighted average strategy: FLAIR (0.4), T1w (0.2), T1wCE (0.2), T2w (0.2)
    print("\nWorkflow: Aggregating predictions via ensemble...")
    final_test_preds = ensemble(all_modality_oof, all_modality_test, y_train_all)
    
    # 5. Compute prediction statistics for reporting
    # We compute an aggregated OOF using the same modality weights
    modality_weights = {'FLAIR': 0.4, 'T1w': 0.2, 'T1wCE': 0.2, 'T2w': 0.2}
    agg_oof = np.zeros_like(y_train_all, dtype=np.float64)
    for name, oof in all_modality_oof.items():
        weight = modality_weights.get(name, 1.0 / len(modalities))
        agg_oof += oof.astype(np.float64) * weight
    
    # 6. Generate final submission file
    submission_df = pd.DataFrame({
        'BraTS21ID': [str(pid).zfill(5) for pid in test_ids],
        'MGMT_value': final_test_preds
    })
    submission_file_path = os.path.join(OUTPUT_DATA_PATH, "submission.csv")
    submission_df.to_csv(submission_file_path, index=False)
    
    # 7. Prepare return metadata
    prediction_stats = {
        "oof": {
            "mean": float(np.mean(agg_oof)),
            "std": float(np.std(agg_oof)),
            "min": float(np.min(agg_oof)),
            "max": float(np.max(agg_oof)),
        },
        "test": {
            "mean": float(np.mean(final_test_preds)),
            "std": float(np.std(final_test_preds)),
            "min": float(np.min(final_test_preds)),
            "max": float(np.max(final_test_preds)),
        }
    }
    
    output_info = {
        "submission_file_path": submission_file_path,
        "prediction_stats": prediction_stats
    }
    
    print(f"Workflow: Pipeline completed successfully. Submission saved to {submission_file_path}")
    return output_info