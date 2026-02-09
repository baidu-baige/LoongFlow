import os
import torch
import pandas as pd
import numpy as np
import joblib
from typing import Tuple, Any

# Pipeline configuration constants
BASE_DATA_PATH = "/mnt/pfs/loongflow/devmachine/h20-02/evolux/output/mlebench/herbarium-2020-fgvc7/prepared/public"
OUTPUT_DATA_PATH = "output/cf84c6d3-8647-45ca-b9ff-02e7ed67cf5b/1/executor/output"

# Task-adaptive type definitions
X = pd.DataFrame  # Feature matrix containing file paths and taxonomic metadata
y = torch.Tensor  # Target vector as PyTorch LongTensor

def preprocess(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame
) -> Tuple[X, y, X, y, X]:
    """
    Transforms raw herbarium data into a model-ready format.

    This implementation:
      1. Maps category_id to a contiguous integer range [0, N-1] for neural network training.
      2. Persists the mapping for downstream inference.
      3. Standardizes feature DataFrames to include taxonomic IDs and transformation instructions.
      4. Implements the technical specification for resizing and augmentation parameters.
    """
    print("Starting preprocessing for Herbarium 2020 dataset...")
    
    # Ensure the output directory exists for persisting transformers/mappings
    os.makedirs(OUTPUT_DATA_PATH, exist_ok=True)

    # Step 1: Target Label Encoding (Species/category_id)
    # We fit the mapping on training data only to maintain strict validation integrity.
    print("Encoding species labels...")
    unique_categories = sorted(y_train.unique())
    cat_to_idx = {cat: i for i, cat in enumerate(unique_categories)}
    idx_to_cat = {i: cat for cat, i in cat_to_idx.items()}
    
    num_classes = len(unique_categories)
    print(f"Detected {num_classes} unique species in training set.")

    # Save mapping for use in the 'ensemble' and 'workflow' stages
    mapping_path = os.path.join(OUTPUT_DATA_PATH, "category_mapping.joblib")
    joblib.dump({'cat_to_idx': cat_to_idx, 'idx_to_cat': idx_to_cat}, mapping_path)

    # Convert targets to PyTorch Tensors
    y_train_processed = torch.tensor(y_train.map(cat_to_idx).values, dtype=torch.long)
    y_val_processed = torch.tensor(y_val.map(cat_to_idx).values, dtype=torch.long)

    # Step 2: Feature Engineering & Transformation Specification
    # For large-scale CV tasks, we prepare the metadata and transform instructions 
    # to be executed by a GPU-accelerated DataLoader in the training stage.
    def build_feature_set(df: pd.DataFrame) -> pd.DataFrame:
        """
        Structures the feature matrix with paths, taxonomy, and preprocessing parameters.
        """
        processed = pd.DataFrame(index=df.index)
        
        # Core image file path
        processed['file_name'] = df['file_name']
        
        # Taxonomic hierarchy (Genus and Family labels)
        # These are used for multi-task learning or hierarchical loss.
        processed['genus_id'] = df['genus_id'] if 'genus_id' in df.columns else -1
        processed['family_id'] = df['family_id'] if 'family_id' in df.columns else -1
        
        # Strategy Implementation: Technical Specification Parameters
        # We embed these into the feature matrix to ensure identical processing across folds.
        processed['resize_height'] = 448
        processed['resize_width'] = 448
        processed['aug_num_ops'] = 2
        processed['aug_magnitude'] = 9
        processed['horizontal_flip_p'] = 0.5
        
        # ImageNet normalization constants (Mean/Std)
        processed['norm_mean'] = str([0.485, 0.456, 0.406])
        processed['norm_std'] = str([0.229, 0.224, 0.225])
        
        return processed

    print("Building processed feature matrices...")
    X_train_processed = build_feature_set(X_train)
    X_val_processed = build_feature_set(X_val)
    X_test_processed = build_feature_set(X_test)

    # Step 3: Validation and Consistency Checks
    # Verify row alignment
    if len(X_train_processed) != len(y_train_processed):
        raise ValueError(f"Train features ({len(X_train_processed)}) and targets ({len(y_train_processed)}) misaligned.")
    if len(X_val_processed) != len(y_val_processed):
        raise ValueError(f"Val features ({len(X_val_processed)}) and targets ({len(y_val_processed)}) misaligned.")
    if len(X_test_processed) != len(X_test):
        raise ValueError("Test set coverage is incomplete.")

    # Verify column consistency
    if not (X_train_processed.columns.equals(X_val_processed.columns) and 
            X_val_processed.columns.equals(X_test_processed.columns)):
        raise ValueError("Feature structure mismatch between train/val/test sets.")

    # Check for invalid values
    if X_train_processed.isnull().any().any():
        raise ValueError("NaN values detected in processed features.")

    print("Preprocessing complete. Data is ready for model training.")
    
    return (
        X_train_processed, 
        y_train_processed, 
        X_val_processed, 
        y_val_processed, 
        X_test_processed
    )