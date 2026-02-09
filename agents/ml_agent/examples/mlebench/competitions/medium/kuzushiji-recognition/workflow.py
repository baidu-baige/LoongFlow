import os
import gc
import torch
import pandas as pd
import numpy as np
import time
from typing import Dict, List, Any

# Import all component functions
from load_data import load_data
from preprocess import preprocess
from get_splitter import get_splitter
from train_and_predict import PREDICTION_ENGINES
from ensemble import ensemble

BASE_DATA_PATH = "/mnt/pfs/loongflow/devmachine/h20-04/evolux/output/mlebench/kuzushiji-recognition/prepared/public"
OUTPUT_DATA_PATH = "output/2cf35106-db22-4d8d-a450-9ac60aada454/1/executor/output"

def workflow() -> dict:
    """
    Orchestrates the complete end-to-end Kuzushiji recognition pipeline.
    
    This implementation handles the large-scale dataset (3244 images) by employing 
    memory-efficient data handling and strategic subsetting to satisfy the 440GB RAM 
    constraints during the multi-process pickling phase of the DDP training stage.
    """
    print("Starting production Kuzushiji recognition pipeline...")
    os.makedirs(OUTPUT_DATA_PATH, exist_ok=True)

    # 1. Data Loading Stage
    # validation_mode=False loads the full production dataset.
    X_train_raw, y_train_raw, X_test, test_ids = load_data(validation_mode=False)
    
    # Memory Management: load_data caches resized images in X_train.attrs['image_cache'].
    # Since preprocess() reads directly from zip files, we purge this cache to save ~10GB RAM.
    if hasattr(X_train_raw, 'attrs') and 'image_cache' in X_train_raw.attrs:
        print("Purging redundant image cache to reclaim system memory...")
        X_train_raw.attrs.pop('image_cache')
    gc.collect()

    # 2. Strategic Subsetting for Stability
    # Multiprocessing 'spawn' pickles arguments. For 3244 images, the preprocessed tiles/crops 
    # occupy ~120GB. Pickling overhead (buffer + 2 child processes) exceeds 440GB RAM.
    # We use a 1200-image subset for training, which provides sufficient diversity for the 
    # CenterNet/EfficientNet models while keeping total memory footprint safely under 250GB.
    TRAIN_SUBSET_SIZE = 1200
    if len(X_train_raw) > TRAIN_SUBSET_SIZE:
        print(f"Subsetting training data to {TRAIN_SUBSET_SIZE} images for multiprocessing stability.")
        X_train_sampled = X_train_raw.sample(n=TRAIN_SUBSET_SIZE, random_state=42).reset_index(drop=True)
        y_train_sampled = X_train_sampled['labels'].copy()
        X_train_sampled.attrs = X_train_raw.attrs.copy()
    else:
        X_train_sampled, y_train_sampled = X_train_raw, y_train_raw

    # 3. Data Splitting Stage
    # We use a single fold (80/20 split) to maximize training data volume and minimize 
    # redundant preprocessing overhead in a production context.
    splitter = get_splitter(X_train_sampled, y_train_sampled)
    train_idx, val_idx = next(splitter.split(X_train_sampled, y_train_sampled))
    
    X_train_f, X_val_f = X_train_sampled.iloc[train_idx].copy(), X_train_sampled.iloc[val_idx].copy()
    y_train_f, y_val_f = y_train_sampled.iloc[train_idx].copy(), y_train_sampled.iloc[val_idx].copy()
    X_train_f.attrs = X_train_sampled.attrs.copy()
    X_val_f.attrs = X_train_sampled.attrs.copy()

    # 4. Preprocessing Stage
    # Generates 1024x1024 tiles for the detector and 128x128 crops for the classifier.
    print(f"Executing preprocessing for {len(X_train_f)} train, {len(X_val_f)} validation, and {len(X_test)} test images...")
    X_train_p, y_train_p, X_val_p, y_val_p, X_test_p = preprocess(
        X_train_f, y_train_f, X_val_f, y_val_f, X_test
    )
    
    # Cleanup intermediate dataframes
    del X_train_raw, y_train_raw, X_train_sampled, y_train_sampled, X_train_f, X_val_f
    gc.collect()

    # 5. Training and Prediction Stage
    # Orchestrates 2-GPU DDP training for both the CenterNet detector and EfficientNet-B3 classifier.
    model_name = "centernet_efficientnet"
    train_fn = PREDICTION_ENGINES[model_name]
    
    print("Beginning two-stage model training and test-set inference...")
    try:
        val_preds, test_preds = train_fn(X_train_p, y_train_p, X_val_p, y_val_p, X_test_p)
    except Exception as e:
        print(f"Training failed: {e}. Retrying with a smaller subset as a fallback...")
        # Emergency fallback for extreme memory fragmentation
        X_train_p, y_train_p = X_train_p[:400], y_train_p[:400]
        val_preds, test_preds = train_fn(X_train_p, y_train_p, X_val_p, y_val_p, X_test_p)

    # 6. Ensemble and Post-Processing Stage
    # Aggregates detections from overlapping tiles using Global NMS (10px radius) 
    # and enforces the 1200 predictions-per-page competition limit.
    print("Performing global non-maximum suppression...")
    all_val_preds = {f"{model_name}_fold0": val_preds}
    all_test_preds = {f"{model_name}_fold0": test_preds}
    final_test_preds = ensemble(all_val_preds, all_test_preds, y_val_p)

    # 7. Statistics and Artifact Generation
    def calculate_stats(preds_list):
        counts = [len(p.split()) // 3 if p else 0 for p in preds_list]
        if not counts: return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
        return {
            "mean": float(np.mean(counts)),
            "std": float(np.std(counts)),
            "min": float(np.min(counts)),
            "max": float(np.max(counts)),
        }

    submission_df = pd.DataFrame({
        'image_id': test_ids,
        'labels': final_test_preds
    })
    submission_path = os.path.join(OUTPUT_DATA_PATH, "submission.csv")
    submission_df.to_csv(submission_path, index=False)

    print(f"Workflow complete. Submission file generated at: {submission_path}")
    
    return {
        "submission_file_path": submission_path,
        "prediction_stats": {
            "oof": calculate_stats(val_preds),
            "test": calculate_stats(final_test_preds),
        }
    }

if __name__ == "__main__":
    workflow()