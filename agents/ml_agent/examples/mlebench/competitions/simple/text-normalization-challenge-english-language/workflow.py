import pandas as pd
import numpy as np
import os
import pickle
from load_data import load_data
from preprocess import preprocess
from get_splitter import get_splitter
from train_and_predict import PREDICTION_ENGINES
from ensemble import ensemble

BASE_DATA_PATH = "/mnt/pfs/loongflow/devmachine/h20-04/evolux/output/mlebench/text-normalization-challenge-english-language/prepared/public"
OUTPUT_DATA_PATH = "output/9fef8e79-9e97-4657-be88-07dd4ac6f366/1/executor/output"

def workflow() -> dict:
    """
    Orchestrates the complete end-to-end machine learning pipeline in production mode.
    
    This implementation executes the pipeline on the full dataset, utilizing GroupKFold 
    cross-validation for robust training and majority-voting fold aggregation. It 
    specifically addresses the index alignment issue between preprocessed DataFrames 
    and numpy target arrays to ensure stability across 9M tokens.
    """
    print("Initiating production workflow for English Text Normalization Challenge...")
    
    # 1. Load full dataset (Production mode: validation_mode=False)
    # The dataset contains approximately 9 million tokens.
    print(f"Loading data from {BASE_DATA_PATH}...")
    X_train, y_train, X_test, test_ids = load_data(validation_mode=False)
    
    # 2. Set up data splitting strategy
    # GroupKFold ensures that all tokens from a sentence remain together in the same fold.
    print("Configuring cross-validation splitter...")
    splitter = get_splitter(X_train, y_train)
    
    # Repositories for predictions across different model engines (if multiple are registered)
    all_oof_preds = {}
    all_combined_test_preds = {}
    
    # 3. Iterate through available prediction engines (e.g., bert_transformer_dual)
    for engine_name, engine_fn in PREDICTION_ENGINES.items():
        print(f"Processing Engine: {engine_name}")
        
        # oof_preds will hold validation predictions for the entire training set
        oof_preds = np.empty(len(X_train), dtype=object)
        test_preds_by_fold = []
        
        # Execute 5-Fold Cross-Validation
        for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(X_train, y_train)):
            print(f"--- Executing Fold {fold_idx + 1} / 5 ---")
            
            # Divide data into training and validation subsets for the current fold
            X_tr, y_tr = X_train.iloc[train_idx], y_train.iloc[train_idx]
            X_va, y_va = X_train.iloc[val_idx], y_train.iloc[val_idx]
            
            # Apply preprocessing
            # Preprocess logic constructs BERT-aligned tokens and character sequences.
            X_tr_p, y_tr_p, X_va_p, y_va_p, X_te_p = preprocess(X_tr, y_tr, X_va, y_va, X_test)
            
            # CRITICAL: Reset indices of preprocessed DataFrames.
            # Preprocess returns y as 0-indexed numpy arrays, but X keeps original Series indices.
            # Resetting X indices ensures 'norm_train.index' aligns with the numpy array positions
            # inside the train_and_predict engine, preventing out-of-bounds errors.
            X_tr_p = X_tr_p.reset_index(drop=True)
            X_va_p = X_va_p.reset_index(drop=True)
            X_te_p = X_te_p.reset_index(drop=True)
            
            # Train and perform inference
            # This utilizes available GPUs for BERT classification and Transformer sequence generation.
            val_preds_fold, test_preds_fold = engine_fn(X_tr_p, y_tr_p, X_va_p, y_va_p, X_te_p)
            
            # Map validation predictions back to original global indices for OOF evaluation
            oof_preds[val_idx] = val_preds_fold
            test_preds_by_fold.append(test_preds_fold)
            
            print(f"Completed Fold {fold_idx + 1}.")

        # Aggregate test predictions across folds via Majority Voting for this specific engine
        print(f"Aggregating fold-wise predictions for {engine_name}...")
        test_matrix = np.array(test_preds_by_fold, dtype=object)
        num_test_samples = test_matrix.shape[1]
        voted_test_preds = np.empty(num_test_samples, dtype=object)
        
        for i in range(num_test_samples):
            votes = test_matrix[:, i]
            unique_vals, counts = np.unique(votes, return_counts=True)
            # Choose the most frequent prediction across folds
            voted_test_preds[i] = unique_vals[np.argmax(counts)]
            
        all_oof_preds[engine_name] = oof_preds
        all_combined_test_preds[engine_name] = voted_test_preds

    # 4. Ensemble predictions from all engines
    # Uses the hierarchical fallback strategy defined in the ensemble component.
    print("Executing final ensemble to produce master predictions...")
    final_test_preds = ensemble(all_oof_preds, all_combined_test_preds, y_train.values)
    
    # 5. Compute Prediction Statistics
    # Since predictions are strings, we compute statistics on string length as a numeric proxy.
    def calculate_stats(predictions):
        lengths = np.array([len(str(p)) for p in predictions])
        return {
            "mean": float(np.mean(lengths)),
            "std": float(np.std(lengths)),
            "min": float(np.min(lengths)),
            "max": float(np.max(lengths))
        }

    # Use the primary engine's OOF for Out-of-Fold statistics
    primary_engine = list(PREDICTION_ENGINES.keys())[0]
    prediction_stats = {
        "oof": calculate_stats(all_oof_preds[primary_engine]),
        "test": calculate_stats(final_test_preds)
    }
    
    # 6. Generate Deliverables and save artifacts
    # Ensure output directory exists before writing
    os.makedirs(OUTPUT_DATA_PATH, exist_ok=True)
    submission_file_path = os.path.join(OUTPUT_DATA_PATH, "submission.csv")
    
    submission_df = pd.DataFrame({
        'id': test_ids.values,
        'after': final_test_preds
    })
    
    # Write final submission CSV according to competition requirements
    submission_df.to_csv(submission_file_path, index=False)
    
    # Persist large prediction metadata for auditability
    with open(os.path.join(OUTPUT_DATA_PATH, "oof_predictions_registry.pkl"), "wb") as f:
        pickle.dump(all_oof_preds, f)

    print(f"Workflow completed. Final submission saved to: {submission_file_path}")
    return {
        "submission_file_path": submission_file_path,
        "prediction_stats": prediction_stats
    }