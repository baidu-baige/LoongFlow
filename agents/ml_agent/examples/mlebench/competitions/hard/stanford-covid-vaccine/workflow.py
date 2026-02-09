import os
import numpy as np
import pandas as pd
from load_data import load_data
from preprocess import preprocess
from get_splitter import get_splitter
from train_and_predict import PREDICTION_ENGINES
from ensemble import ensemble

# Define paths relative to the environment setup
BASE_DATA_PATH = "/mnt/pfs/loongflow/devmachine/h20-11/evolux/output/mlebench/stanford-covid-vaccine/prepared/public"
OUTPUT_DATA_PATH = "output/cd1762a7-cbef-43b4-bbfc-bc919a0a7546/1/executor/output"

def workflow() -> dict:
    """
    Orchestrates the complete end-to-end machine learning pipeline in production mode.
    """
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DATA_PATH, exist_ok=True)

    # 1. Load full dataset
    print("Step 1: Loading data...")
    X_train_raw, y_train_raw, X_test_raw, test_ids = load_data(validation_mode=False)
    
    # 2. Set up data splitting strategy
    print("Step 2: Initializing splitter...")
    splitter = get_splitter(X_train_raw, y_train_raw)
    n_splits = splitter.get_n_splits()
    
    # Containers for cross-validation results
    model_names = list(PREDICTION_ENGINES.keys())
    # OOF predictions: (N_train, 107, 5) for each model
    all_oof_preds = {name: np.zeros((len(X_train_raw), 107, 5)) for name in model_names}
    # Test predictions: (N_test, 107, 5) for each model, averaged over folds
    all_test_preds_avg = {name: np.zeros((len(X_test_raw), 107, 5)) for name in model_names}
    # Combined ground truth for the whole training set (following SN_filter)
    y_true_full = np.zeros((len(X_train_raw), 107, 5))
    
    # 3. K-Fold Cross-Validation
    print(f"Step 3: Starting {n_splits}-fold cross-validation...")
    for fold, (train_idx, val_idx) in enumerate(splitter.split(X_train_raw, y_train_raw)):
        print(f"\n--- Processing Fold {fold + 1}/{n_splits} ---")
        
        # Split raw data for this fold
        X_train_f, X_val_f = X_train_raw.iloc[train_idx], X_train_raw.iloc[val_idx]
        y_train_f, y_val_f = y_train_raw.iloc[train_idx], y_train_raw.iloc[val_idx]
        
        # Preprocess features and targets into aligned arrays
        X_train_p, y_train_p, X_val_p, y_val_p, X_test_p = preprocess(
            X_train_f, y_train_f, X_val_f, y_val_f, X_test_raw
        )
        
        # Store processed ground truth for validation metrics
        y_true_full[val_idx] = y_val_p
        
        # Train and Predict for each available engine
        for name, engine_fn in PREDICTION_ENGINES.items():
            print(f"Training engine: {name} on Fold {fold + 1}")
            val_preds, test_preds = engine_fn(X_train_p, y_train_p, X_val_p, y_val_p, X_test_p)
            
            # Aggregate predictions
            all_oof_preds[name][val_idx] = val_preds
            all_test_preds_avg[name] += test_preds / n_splits

    # 4. Ensemble predictions from all models
    # The ensemble function computes simple average and calculates MCRMSE on 68 bases
    print("\nStep 4: Ensembling predictions...")
    final_test_preds = ensemble(all_oof_preds, all_test_preds_avg, y_true_full)
    
    # 5. Compute prediction statistics
    print("Step 5: Computing prediction statistics...")
    # Calculate OOF stats by averaging all models (if multiple)
    combined_oof = np.mean(list(all_oof_preds.values()), axis=0)
    
    stats = {
        "oof": {
            "mean": float(np.mean(combined_oof)),
            "std": float(np.std(combined_oof)),
            "min": float(np.min(combined_oof)),
            "max": float(np.max(combined_oof)),
        },
        "test": {
            "mean": float(np.mean(final_test_preds)),
            "std": float(np.std(final_test_preds)),
            "min": float(np.min(final_test_preds)),
            "max": float(np.max(final_test_preds)),
        }
    }
    
    # 6. Generate submission file
    # Format: id_seqpos, reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C
    print("Step 6: Formatting submission file...")
    target_cols = ['reactivity', 'deg_Mg_pH10', 'deg_pH10', 'deg_Mg_50C', 'deg_50C']
    submission_rows = []
    
    # Align final_test_preds with test_ids and sequence length
    for i, sample_id in enumerate(test_ids):
        # Determine sequence length for this specific sample
        seq_len = int(X_test_raw.iloc[i]['seq_length'])
        for pos in range(seq_len):
            row = {'id_seqpos': f"{sample_id}_{pos}"}
            for t_idx, col in enumerate(target_cols):
                # final_test_preds shape is (N, 107, 5). 
                # If seq_len matches 107, index directly. 
                # Note: The technical spec and preprocess assume max 107.
                row[col] = final_test_preds[i, pos, t_idx]
            submission_rows.append(row)
    
    submission_df = pd.DataFrame(submission_rows)
    submission_path = os.path.join(OUTPUT_DATA_PATH, "submission.csv")
    submission_df.to_csv(submission_path, index=False)
    print(f"Submission saved to: {submission_path}")

    # 7. Return required output info
    output_info = {
        "submission_file_path": submission_path,
        "prediction_stats": stats,
    }
    
    print("Workflow complete.")
    return output_info

if __name__ == "__main__":
    workflow()