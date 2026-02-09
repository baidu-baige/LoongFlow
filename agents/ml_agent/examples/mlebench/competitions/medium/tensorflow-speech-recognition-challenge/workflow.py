import os
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score

# import all component functions
from load_data import load_data
from preprocess import preprocess
from get_splitter import get_splitter
from train_and_predict import PREDICTION_ENGINES
from ensemble import ensemble

BASE_DATA_PATH = "/mnt/pfs/loongflow/devmachine/2-05/evolux/output/mlebench/tensorflow-speech-recognition-challenge/prepared/public"
OUTPUT_DATA_PATH = "output/fe927f25-451a-41da-a547-cdb392b784d8/1/executor/output"

def workflow() -> dict:
    """
    Orchestrates the complete end-to-end machine learning pipeline in production mode.
    """
    print("Starting full production pipeline workflow...")

    # Ensure output directory exists
    os.makedirs(OUTPUT_DATA_PATH, exist_ok=True)

    # 1. Load full dataset
    # Returns X_train (path, speaker_id), y_train (label), X_test (path), test_ids (fname)
    X_train_full, y_train_full, X_test_df, test_ids = load_data(validation_mode=False)
    
    # 2. Set up data splitting strategy
    # Uses GroupKFold based on speaker_id
    splitter = get_splitter(X_train_full, y_train_full)
    n_folds = splitter.get_n_splits(X_train_full, y_train_full)
    
    # Storage for CV results
    num_classes = 31 # 30 words + silence
    oof_probs = np.zeros((len(X_train_full), num_classes))
    all_test_preds_folds = []
    
    # Reconstruct the fixed alphabetical class list for consistent encoding/decoding
    train_audio_dir = os.path.join(BASE_DATA_PATH, "train/audio")
    word_labels = sorted([
        d for d in os.listdir(train_audio_dir)
        if os.path.isdir(os.path.join(train_audio_dir, d)) and not d.startswith('_')
    ])
    all_classes = sorted(word_labels + ['silence'])
    label_to_idx = {label: i for i, label in enumerate(all_classes)}
    y_train_encoded_full = np.array([label_to_idx[l] for l in y_train_full])

    # 3. Cross-Validation Loop
    print(f"Executing {n_folds}-fold cross-validation...")
    for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(X_train_full, y_train_full)):
        print(f"--- Processing Fold {fold_idx + 1}/{n_folds} ---")
        
        # Split data for this fold
        X_tr, X_va = X_train_full.iloc[train_idx], X_train_full.iloc[val_idx]
        y_tr, y_va = y_train_full.iloc[train_idx], y_train_full.iloc[val_idx]
        
        # a. Preprocess (Waveform to Log-Mel Spectrogram)
        # Note: X_test is processed in every fold to allow for potential fold-specific variations or weighting
        X_tr_p, y_tr_p, X_va_p, y_va_p, X_te_p = preprocess(X_tr, y_tr, X_va, y_va, X_test_df)
        
        # b. Train and Predict
        # Using the EfficientNet-B0 engine as registered
        model_name = "efficientnet_b0"
        train_fn = PREDICTION_ENGINES[model_name]
        
        # returns softmax probabilities (N, 31)
        val_preds, test_preds = train_fn(X_tr_p, y_tr_p, X_va_p, y_va_p, X_te_p)
        
        # Store predictions
        oof_probs[val_idx] = val_preds
        all_test_preds_folds.append(test_preds)
        
        # Clean up large arrays to manage memory
        del X_tr_p, y_tr_p, X_va_p, y_va_p, X_te_p
        
    # 4. Ensemble and Final Prediction
    print("Ensembling fold predictions...")
    all_val_preds_dict = {"efficientnet_b0_oof": oof_probs}
    all_test_preds_dict = {f"fold_{i}": p for i, p in enumerate(all_test_preds_folds)}
    
    # ensemble() performs soft-voting on test_preds and maps indices back to labels
    final_test_labels = ensemble(
        all_val_preds=all_val_preds_dict,
        all_test_preds=all_test_preds_dict,
        y_val=y_train_encoded_full
    )
    
    # 5. Compute Metrics and Statistics
    # Calculate OOF Accuracy
    oof_indices = np.argmax(oof_probs, axis=1)
    oof_accuracy = accuracy_score(y_train_encoded_full, oof_indices)
    print(f"Overall OOF Accuracy: {oof_accuracy:.4f}")
    
    # Calculate Mean Test Probabilities for stats
    avg_test_probs = np.mean(all_test_preds_folds, axis=0)
    
    prediction_stats = {
        "oof": {
            "mean": float(np.mean(oof_probs)),
            "std": float(np.std(oof_probs)),
            "min": float(np.min(oof_probs)),
            "max": float(np.max(oof_probs)),
        },
        "test": {
            "mean": float(np.mean(avg_test_probs)),
            "std": float(np.std(avg_test_probs)),
            "min": float(np.min(avg_test_probs)),
            "max": float(np.max(avg_test_probs)),
        }
    }

    # 6. Generate Deliverables
    submission_path = os.path.join(OUTPUT_DATA_PATH, "submission.csv")
    submission_df = pd.DataFrame({
        "fname": test_ids,
        "label": final_test_labels
    })
    submission_df.to_csv(submission_path, index=False)
    print(f"Submission file saved to {submission_path}")
    
    # Prepare final output info
    output_info = {
        "submission_file_path": submission_path,
        "model_scores": {
            "efficientnet_b0_cv_accuracy": float(oof_accuracy)
        },
        "prediction_stats": prediction_stats,
    }
    
    print("Workflow execution completed successfully.")
    return output_info