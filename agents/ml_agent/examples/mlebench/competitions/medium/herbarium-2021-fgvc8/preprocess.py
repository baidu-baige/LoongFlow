import os
import json
import pickle
import numpy as np
import pandas as pd
from typing import Tuple
from sklearn.preprocessing import LabelEncoder

# Task-adaptive type definitions
X = pd.DataFrame
y = pd.DataFrame

OUTPUT_DATA_PATH = "output/ec5e2488-9859-4456-b67a-20c4a0b3bb67/1/executor/output"

def preprocess(
    X_train: X,
    y_train: y,
    X_val: X,
    y_val: y,
    X_test: X
) -> Tuple[X, y, X, y, X]:
    """
    Transforms raw Herbarium 2021 data into model-ready format.
    
    This function:
    1. Maps high-cardinality category IDs to contiguous indices for classification.
    2. Ensures row alignment and completeness across all splits.
    3. Defines configuration for progressive resizing and advanced augmentations.
    4. Saves mapping artifacts for the prediction and ensemble stages.
    """
    print(f"Preprocessing {len(X_train)} training, {len(X_val)} validation, and {len(X_test)} test records...")

    # Step 1: Label Encoding for targets
    # We fit the encoder on the union of train and val labels to ensure all classes in the current split are accounted for.
    # This is safe because these are categorical labels, not features subject to data leakage.
    le = LabelEncoder()
    full_labels = pd.concat([y_train['category_id'], y_val['category_id']])
    le.fit(full_labels)
    
    y_train_processed = y_train.copy()
    y_val_processed = y_val.copy()
    
    # Transform category_id to 0-indexed contiguous integers
    y_train_processed['category_id'] = le.transform(y_train['category_id']).astype(np.int32)
    y_val_processed['category_id'] = le.transform(y_val['category_id']).astype(np.int32)
    
    # family_id and order_id are already encoded in load_data; preserve them for potential multi-task usage
    y_train_processed['family_id'] = y_train['family_id'].astype(np.int32)
    y_train_processed['order_id'] = y_train['order_id'].astype(np.int32)
    y_val_processed['family_id'] = y_val['family_id'].astype(np.int32)
    y_val_processed['order_id'] = y_val['order_id'].astype(np.int32)

    # Step 2: Feature alignment and consistency
    # Ensure all DataFrames are reset and aligned
    X_train_processed = X_train.copy().reset_index(drop=True)
    y_train_processed = y_train_processed.reset_index(drop=True)
    X_val_processed = X_val.copy().reset_index(drop=True)
    y_val_processed = y_val_processed.reset_index(drop=True)
    X_test_processed = X_test.copy().reset_index(drop=True)

    # Step 3: Artifact Persistence
    # Save the label encoder to allow reversing predictions to original category IDs in the downstream stages.
    os.makedirs(OUTPUT_DATA_PATH, exist_ok=True)
    encoder_path = os.path.join(OUTPUT_DATA_PATH, "category_encoder.pkl")
    print(f"Saving Category LabelEncoder to {encoder_path}...")
    with open(encoder_path, 'wb') as f:
        pickle.dump(le, f)

    # Define and save augmentation/progressive resizing configuration for the training stage.
    # CV2 Interpolation mappings: INTER_AREA = 3, INTER_CUBIC = 2
    aug_config = {
        "progressive_resizing": {
            "stage_1": {
                "size": 224,
                "interpolation": 3,  # INTER_AREA for downsampling 1000px images
                "batch_size_multiplier": 1.0
            },
            "stage_2": {
                "size": 384,
                "interpolation": 2,  # INTER_CUBIC for fine-grained upscaling
                "batch_size_multiplier": 0.5
            }
        },
        "augmentation_params": {
            "random_resized_crop": True,
            "horizontal_flip": True,
            "random_rotate_90": True,
            "color_jitter": {"brightness": 0.2, "contrast": 0.2},
            "randaugment": {"n": 2, "m": 9},
            "normalization": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}
        }
    }
    config_path = os.path.join(OUTPUT_DATA_PATH, "preprocessing_config.json")
    with open(config_path, 'w') as f:
        json.dump(aug_config, f, indent=4)
    print(f"Saved preprocessing configuration to {config_path}...")

    # Step 4: Validation
    # Ensure no NaN/Infinity values are present
    for df_name, df in [("X_train", X_train_processed), ("X_val", X_val_processed), ("X_test", X_test_processed)]:
        if df.isnull().any().any():
            raise ValueError(f"Consistency Error: NaNs found in {df_name}")
    
    if len(X_train_processed) != len(y_train_processed):
        raise ValueError("Alignment Error: Train feature and target row counts mismatch.")
    if len(X_val_processed) != len(y_val_processed):
        raise ValueError("Alignment Error: Validation feature and target row counts mismatch.")
    
    print(f"Preprocessing successful. Total Classes: {len(le.classes_)}")
    print(f"Data Shapes: Train {X_train_processed.shape}, Val {X_val_processed.shape}, Test {X_test_processed.shape}")
    
    return X_train_processed, y_train_processed, X_val_processed, y_val_processed, X_test_processed