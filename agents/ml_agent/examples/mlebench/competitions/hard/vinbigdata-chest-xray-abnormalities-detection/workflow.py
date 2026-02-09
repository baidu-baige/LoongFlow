import os
import pandas as pd
import numpy as np
import torch
import gc
from load_data import load_data
from preprocess import preprocess
from get_splitter import get_splitter
from train_and_predict import PREDICTION_ENGINES
from ensemble import ensemble

BASE_DATA_PATH = "/mnt/pfs/loongflow/devmachine/h20-10/evolux/output/mlebench/vinbigdata-chest-xray-abnormalities-detection/prepared/public"
OUTPUT_DATA_PATH = "output/7357c439-e7eb-4cf1-afb0-36b77c92c672/1/executor/output"

def workflow() -> dict:
    """
    Orchestrates the complete VinBigData Chest X-ray detection pipeline.
    Optimized for memory efficiency to avoid CUDA Out-Of-Memory errors.
    """
    print("Stage 1: Initializing VinBigData Production Pipeline...")
    # Explicitly clear any existing GPU memory
    torch.cuda.empty_cache()
    gc.collect()
    
    # 1. Load full dataset
    X_train_full, y_train_full, X_test, test_ids = load_data(validation_mode=False)
    
    # 2. Preprocess all data once to avoid redundant computations and spikes in GPU memory
    print("Stage 2: Preprocessing all images (Train + Test) once...")
    # We pass a single sample for validation to satisfy the function signature
    X_train_p_all, y_train_p_all, _, _, X_test_p = preprocess(
        X_train_full, y_train_full, 
        X_train_full.head(1), [y_train_full[0]], 
        X_test
    )
    # Clear cache after preprocessing
    torch.cuda.empty_cache()
    gc.collect()

    # 3. Set up data splitting strategy
    splitter = get_splitter(X_train_full, y_train_full)
    
    all_val_preds = {}
    all_test_preds = {}
    oof_predictions = [None] * len(X_train_full)
    
    # Use the 2-stage engine specified in technical requirements
    engine = PREDICTION_ENGINES["effnet_yolov11_2stage"]
    
    # 4. Cross-Validation Loop
    num_folds = splitter.get_n_splits()
    for fold, (train_idx, val_idx) in enumerate(splitter.split(X_train_full, y_train_full)):
        print(f"\n--- Processing Fold {fold + 1}/{num_folds} ---")
        
        # Efficiently slice the preprocessed CPU tensors/lists
        X_tr_p = X_train_p_all[train_idx]
        y_tr_p = [y_train_p_all[i] for i in train_idx]
        X_val_p = X_train_p_all[val_idx]
        y_val_p = [y_train_p_all[i] for i in val_idx]
        
        # Stage 4: Train and Predict (Classifier + Detector)
        # Note: We pass the full X_test_p as it is already preprocessed
        val_preds, test_preds = engine(X_tr_p, y_tr_p, X_val_p, y_val_p, X_test_p)
        
        # Store predictions
        all_val_preds[f"fold_{fold}"] = val_preds
        all_test_preds[f"fold_{fold}"] = test_preds
        
        for i, idx in enumerate(val_idx):
            oof_predictions[idx] = val_preds[i]
            
        # Crucial: Clean up after each fold to free GPU and RAM
        del X_tr_p, y_tr_p, X_val_p, y_val_p
        gc.collect()
        torch.cuda.empty_cache()

    # 5. Ensemble predictions from all folds using Weighted Boxes Fusion (WBF)
    print("\nStage 5: Ensembling fold predictions...")
    final_test_preds_1024 = ensemble(all_val_preds, all_test_preds, y_train_full)
    
    # 6. Post-processing: Rescale bounding boxes from 1024x1024 back to original dimensions
    def rescale_predictions(preds, meta_df):
        rescaled_list = []
        target_size = 1024.0
        for i, pred_str in enumerate(preds):
            # image_metadata_full.csv contains the original width and height
            orig_w = meta_df.iloc[i]['width']
            orig_h = meta_df.iloc[i]['height']
            
            if pred_str == "14 1.0 0 0 1 1":
                rescaled_list.append(pred_str)
                continue
            
            # Recompute Letterbox parameters used in preprocess.py
            scale = target_size / max(orig_h, orig_w)
            new_w, new_h = int(orig_w * scale), int(orig_h * scale)
            pad_x = (target_size - new_w) // 2
            pad_y = (target_size - new_h) // 2
            
            parts = pred_str.split()
            res = []
            for j in range(0, len(parts), 6):
                cls_id = int(parts[j])
                conf = float(parts[j+1])
                # Invert the letterbox scaling and padding
                x1 = (float(parts[j+2]) - pad_x) / scale
                y1 = (float(parts[j+3]) - pad_y) / scale
                x2 = (float(parts[j+4]) - pad_x) / scale
                y2 = (float(parts[j+5]) - pad_y) / scale
                
                # Clip to original image boundaries
                x1, x2 = np.clip([x1, x2], 0, orig_w)
                y1, y2 = np.clip([y1, y2], 0, orig_h)
                
                res.append(f"{cls_id} {conf:.4f} {int(round(x1))} {int(round(y1))} {int(round(x2))} {int(round(y2))}")
            rescaled_list.append(" ".join(res))
        return rescaled_list

    print("Stage 6: Rescaling coordinates to original dimensions...")
    rescaled_test_preds = rescale_predictions(final_test_preds_1024, X_test)
    
    # 7. Generate Final Submission File
    submission = pd.DataFrame({
        'image_id': test_ids,
        'PredictionString': rescaled_test_preds
    })
    
    # Load sample submission to ensure exact image_id order and coverage
    sample_sub_path = os.path.join(BASE_DATA_PATH, "sample_submission.csv")
    sample_sub = pd.read_csv(sample_sub_path)
    final_submission = sample_sub[['image_id']].merge(submission, on='image_id', how='left')
    final_submission['PredictionString'] = final_submission['PredictionString'].fillna("14 1.0 0 0 1 1")
    
    sub_file_path = os.path.join(OUTPUT_DATA_PATH, "submission.csv")
    final_submission.to_csv(sub_file_path, index=False)
    print(f"Submission file created: {sub_file_path}")
    
    # 8. Compute prediction statistics for the return dict
    def get_stats(preds_list):
        scores = []
        for p_str in preds_list:
            if p_str:
                parts = p_str.split()
                for k in range(1, len(parts), 6):
                    scores.append(float(parts[k]))
        if not scores: return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
        return {
            "mean": float(np.mean(scores)),
            "std": float(np.std(scores)),
            "min": float(np.min(scores)),
            "max": float(np.max(scores)),
        }

    prediction_stats = {
        "oof": get_stats(oof_predictions),
        "test": get_stats(rescaled_test_preds)
    }
    
    print("Pipeline execution completed successfully.")
    return {
        "submission_file_path": sub_file_path,
        "prediction_stats": prediction_stats
    }