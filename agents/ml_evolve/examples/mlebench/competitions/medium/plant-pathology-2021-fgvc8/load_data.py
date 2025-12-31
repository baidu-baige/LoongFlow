# -*- coding: utf-8 -*-
"""
LLM Generated Code
"""

import os
from typing import Tuple

import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

BASE_DATA_PATH = "/root/workspace/evolux_ml/output/mlebench/plant-pathology-2021-fgvc8/prepared/public"
OUTPUT_DATA_PATH = "output/1ae93cb9-976c-4242-a6ab-6fcbc92f5502/1/executor/output"

# For type hinting, DT is assumed to be pandas DataFrame or Series
DT = pd.DataFrame | pd.Series


def load_data(validation_mode: bool = False) -> Tuple[DT, DT, DT, DT]:
    """
    Loads, splits, and returns the initial datasets for the Plant Pathology 2021 competition.
    
    Args:
        validation_mode: Controls the data loading behavior.
            - False (default): Load the complete dataset.
            - True: Load a small subset (≤100 rows).

    Returns:
        Tuple[DT, DT, DT, DT]: (X, y, X_test, test_ids)
    """
    train_csv_path = os.path.join(BASE_DATA_PATH, "train.csv")
    test_csv_path = os.path.join(BASE_DATA_PATH, "sample_submission.csv")
    train_images_dir = os.path.join(BASE_DATA_PATH, "train_images")
    test_images_dir = os.path.join(BASE_DATA_PATH, "test_images")

    # Load dataframes
    if validation_mode:
        train_df = pd.read_csv(train_csv_path, nrows=100)
        test_df = pd.read_csv(test_csv_path, nrows=100)
    else:
        train_df = pd.read_csv(train_csv_path)
        test_df = pd.read_csv(test_csv_path)

    # Multi-label transformation
    # Defined base labels from dataset description and EDA
    classes = ['scab', 'healthy', 'frog_eye_leaf_spot', 'rust', 'complex', 'powdery_mildew']
    mlb = MultiLabelBinarizer(classes=classes)

    # Each row in 'labels' is a space-delimited string
    y_labels = train_df['labels'].apply(lambda x: x.split())
    y_encoded = mlb.fit_transform(y_labels)
    y = pd.DataFrame(y_encoded, columns=classes)

    # Features X: Store image filenames (relative to the train_images directory)
    X = train_df[['image']].rename(columns={'image': 'image_path'})

    # Features X_test: Consistent structure with X
    X_test = test_df[['image']].rename(columns={'image': 'image_path'})

    # Identifiers for test data
    test_ids = test_df['image'].copy()

    # Verification of image existence
    # Using set lookup for efficiency
    actual_train_files = set(os.listdir(train_images_dir))
    for img_name in X['image_path']:
        if img_name not in actual_train_files:
            raise FileNotFoundError(f"Training image {img_name} missing from {train_images_dir}")

    actual_test_files = set(os.listdir(test_images_dir))
    for img_name in X_test['image_path']:
        if img_name not in actual_test_files:
            raise FileNotFoundError(f"Test image {img_name} missing from {test_images_dir}")

    # Final consistency checks
    if X.empty or y.empty or X_test.empty or test_ids.empty:
        raise ValueError("One or more returned datasets are empty.")

    if len(X) != len(y):
        raise ValueError("X and y row counts do not match.")

    if len(X_test) != len(test_ids):
        raise ValueError("X_test and test_ids row counts do not match.")

    return X, y, X_test, test_ids
