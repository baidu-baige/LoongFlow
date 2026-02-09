import os
import gc
import pandas as pd
import numpy as np
import pickle
import torch
from typing import Dict, List, Any
from load_data import load_data
from preprocess import preprocess
from get_splitter import get_splitter
from train_and_predict import PREDICTION_ENGINES
from ensemble import ensemble

BASE_DATA_PATH = "/mnt/pfs/loongflow/devmachine/h20-12/evolux/output/mlebench/3d-object-detection-for-autonomous-vehicles/prepared/public"
OUTPUT_DATA_PATH = "output/341879f4-a7a1-4d18-a801-6d16eb36af1b/1/executor/output"

def workflow() -> dict:
    """
    Orchestrates the complete end-to-end machine learning pipeline.
    
    To prevent OOM (Out-of-Memory) errors caused by the massive scale of 3D voxel data 
    and the overhead of pickling large objects during DDP (Distributed Data Parallel) 
    initialization, this workflow utilizes a strategic subset of the training data 
    while maintaining the full test set for the final submission.
    """
    print("Starting pipeline: load_data")
    # 1. Load full dataset
    X_train_raw, y_train_raw, X_test_raw, test_ids = load_data(validation_mode=False)
    
    print("Starting pipeline: get_splitter")
    splitter = get_splitter(X_train_raw, y_train_raw)
    
    # 2. Select a single fold and subset training data to fit in memory
    # Given 440GB RAM and ~10MB per voxelized sample, we aim for ~10,000 total processed samples
    # to allow for the doubling of memory usage during mp.spawn pickling.
    train_idx_full, val_idx_full = next(splitter.split(X_train_raw, y_train_raw))
    
    # Subset train/val to ensure stability
    # We take 2000 training and 500 validation samples to leave room for the 5000 test samples
    # and the CBGS expansion inside the preprocess function.
    train_idx = train_idx_full[:2000]
    val_idx = val_idx_full[:500]
    
    print(f"Subsetting data: {len(train_idx)} train, {len(val_idx)} val, {len(X_test_raw)} test.")
    
    X_tr = [X_train_raw[i] for i in train_idx]
    y_tr = [y_train_raw[i] for i in train_idx]
    X_va = [X_train_raw[i] for i in val_idx]
    y_va = [y_train_raw[i] for i in val_idx]
    
    # Clean up raw data pointers
    del X_train_raw, y_train_raw
    gc.collect()
    
    # 3. Preprocess data (Voxelization, CBGS, Augmentation)
    print("Executing preprocess component...")
    X_tr_p, y_tr_p, X_va_p, y_va_p, X_te_p = preprocess(X_tr, y_tr, X_va, y_va, X_test_raw)
    
    # CBGS might have expanded X_tr_p. If it's still too large (>5000), we cap it.
    if len(X_tr_p) > 5000:
        print(f"Capping CBGS-expanded training set from {len(X_tr_p)} to 5000 for RAM safety.")
        X_tr_p = X_tr_p[:5000]
        y_tr_p = y_tr_p[:5000]
    
    # Memory Optimization: Use float16 for voxels to save 50% RAM
    print("Converting voxel features to float16...")
    for dataset in [X_tr_p, X_va_p, X_te_p]:
        for sample in dataset:
            if 'voxels' in sample:
                sample['voxels'] = sample['voxels'].astype(np.float16)

    # Clean up intermediate slices
    del X_tr, y_tr, X_va, X_test_raw
    gc.collect()
    
    all_val_preds_dict = {}
    all_test_preds_dict = {}
    oof_confidences = []
    
    # 4. Train and Predict
    # Uses 'centerpoint' model which internally handles 2x GPU DDP and inference
    for model_name, train_func in PREDICTION_ENGINES.items():
        print(f"Running training and inference for engine: {model_name}")
        val_preds, test_preds = train_func(X_tr_p, y_tr_p, X_va_p, y_va_p, X_te_p)
        
        all_val_preds_dict[model_name] = val_preds
        all_test_preds_dict[model_name] = test_preds
        
        for boxes in val_preds:
            oof_confidences.extend([b['confidence'] for b in boxes])
            
    # Clean up processed voxel data
    del X_tr_p, y_tr_p, X_va_p, X_te_p
    gc.collect()

    # 5. Ensemble predictions using 3D-NMS
    print("Ensembling predictions...")
    final_test_preds = ensemble(all_val_preds_dict, all_test_preds_dict, y_va)
    
    # 6. Compute Statistics
    test_confidences = []
    for boxes in final_test_preds:
        test_confidences.extend([b['confidence'] for b in boxes])
        
    def calculate_stats(scores):
        if not scores:
            return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
        return {
            "mean": float(np.mean(scores)),
            "std": float(np.std(scores)),
            "min": float(np.min(scores)),
            "max": float(np.max(scores))
        }

    prediction_stats = {
        "oof": calculate_stats(oof_confidences),
        "test": calculate_stats(test_confidences)
    }
    
    # 7. Generate Final Submission
    print("Creating submission.csv...")
    sample_sub = pd.read_csv(os.path.join(BASE_DATA_PATH, "sample_submission.csv"))
    
    # Map predictions to Id
    pred_map = {test_ids[i]: final_test_preds[i] for i in range(len(test_ids))}
    
    submission_data = []
    for _, row in sample_sub.iterrows():
        token = row['Id']
        boxes = pred_map.get(token, [])
        strings = []
        for b in boxes:
            # Format: confidence x y z w l h yaw class
            strings.append(f"{b['confidence']:.4f} {b['center_x']:.4f} {b['center_y']:.4f} {b['center_z']:.4f} "
                           f"{b['width']:.4f} {b['length']:.4f} {b['height']:.4f} {b['yaw']:.4f} {b['class_name']}")
        
        submission_data.append({"Id": token, "PredictionString": " ".join(strings)})
    
    submission_df = pd.DataFrame(submission_data)
    submission_path = os.path.join(OUTPUT_DATA_PATH, "submission.csv")
    submission_df.to_csv(submission_path, index=False)
    
    # Save artifacts
    with open(os.path.join(OUTPUT_DATA_PATH, "prediction_stats.json"), 'wb') as f:
        pickle.dump(prediction_stats, f)

    print(f"Workflow complete. Submission saved at: {submission_path}")
    return {
        "submission_file_path": submission_path,
        "prediction_stats": prediction_stats
    }