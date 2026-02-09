import os
import numpy as np
import pandas as pd
from typing import Tuple

# Task-specific paths
BASE_DATA_PATH = "/mnt/pfs/loongflow/devmachine/h20-02/evolux/output/mlebench/cdiscount-image-classification-challenge/prepared/public"
OUTPUT_DATA_PATH = "output/96ece161-83e4-4d99-a688-ea7a2b1aa242/1/executor/output"

# Task-adaptive type definitions
X = pd.DataFrame      # Metadata index (product IDs, offsets, lengths, BSON paths)
y = pd.DataFrame      # Target indices (category, level1, level2)

def preprocess(
    X_train: X,
    y_train: y,
    X_val: X,
    y_val: y,
    X_test: X
) -> Tuple[X, y, X, y, X]:
    """
    Transforms raw metadata into model-ready format for high-throughput image classification.
    
    This function prepares the metadata (offsets, lengths, paths) required for on-the-fly 
    BSON decoding and image augmentation. It ensures data integrity and memory efficiency
    for the downstream EfficientNet-B1 training stage.
    """
    print("Preprocess stage: Preparing metadata for on-the-fly image decoding...")

    # Define BSON paths for the downstream Dataset loading
    train_bson_path = os.path.join(BASE_DATA_PATH, "train.bson")
    test_bson_path = os.path.join(BASE_DATA_PATH, "test.bson")
    
    # Use copies to avoid SettingWithCopy warnings
    X_train_proc = X_train.copy()
    X_val_proc = X_val.copy()
    X_test_proc = X_test.copy()
    y_train_proc = y_train.copy()
    y_val_proc = y_val.copy()

    # Assign BSON source paths to metadata to allow the DataLoader to find binary data
    X_train_proc['bson_path'] = train_bson_path
    X_val_proc['bson_path'] = train_bson_path
    X_test_proc['bson_path'] = test_bson_path

    # Memory Optimization: Downcast to compact numeric representations
    # Essential for handling the scale of Cdiscount's 15M+ images within RAM.
    def optimize_dtypes(df, is_feature=True):
        if is_feature:
            # Features: _id (int64), img_idx (int8), offset (int64), length (int32)
            df['_id'] = df['_id'].astype(np.int64)
            df['img_idx'] = df['img_idx'].astype(np.int8)
            df['offset'] = df['offset'].astype(np.int64)
            df['length'] = df['length'].astype(np.int32)
        else:
            # Targets: cat_idx (int16), l1_idx (int8), l2_idx (int16)
            if 'cat_idx' in df.columns:
                df['cat_idx'] = df['cat_idx'].astype(np.int16)
            if 'l1_idx' in df.columns:
                df['l1_idx'] = df['l1_idx'].astype(np.int8)
            if 'l2_idx' in df.columns:
                df['l2_idx'] = df['l2_idx'].astype(np.int16)
        return df

    X_train_proc = optimize_dtypes(X_train_proc, is_feature=True)
    X_val_proc = optimize_dtypes(X_val_proc, is_feature=True)
    X_test_proc = optimize_dtypes(X_test_proc, is_feature=True)
    y_train_proc = optimize_dtypes(y_train_proc, is_feature=False)
    y_val_proc = optimize_dtypes(y_val_proc, is_feature=False)

    # Data Integrity Verification: Ensure no NaN or Infinity values exist
    for df_name, df in [("X_train", X_train_proc), ("y_train", y_train_proc), 
                        ("X_val", X_val_proc), ("y_val", y_val_proc), ("X_test", X_test_proc)]:
        if df.isnull().any().any():
            raise ValueError(f"Preprocessing failed: NaN values detected in {df_name}")
        
        # Check for non-finite values in numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if not np.isfinite(df[numeric_cols].values).all():
            raise ValueError(f"Preprocessing failed: Infinity values detected in {df_name}")

    # Row alignment validation
    assert len(X_train_proc) == len(y_train_proc), "Metadata and Target row mismatch in Train"
    assert len(X_val_proc) == len(y_val_proc), "Metadata and Target row mismatch in Val"
    assert len(X_test_proc) == len(X_test), "Test metadata length mismatch"

    # Standardize column structure for downstream consumption
    feature_cols = ['_id', 'img_idx', 'offset', 'length', 'bson_path']
    X_train_proc = X_train_proc[feature_cols]
    X_val_proc = X_val_proc[feature_cols]
    X_test_proc = X_test_proc[feature_cols]

    print(f"Preprocess complete. Ready for training with 5,270 classes.")
    print(f"Technical Specs defined for downstream: EfficientNet-B1, Mixup(0.2), ImageNet norm, cv2.imdecode.")

    return X_train_proc, y_train_proc, X_val_proc, y_val_proc, X_test_proc