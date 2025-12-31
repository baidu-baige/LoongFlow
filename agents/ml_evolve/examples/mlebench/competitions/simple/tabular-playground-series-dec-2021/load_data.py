# -*- coding: utf-8 -*-
"""
LLM Generated Code
"""

import os
from typing import Tuple

import cudf
import pandas as pd

BASE_DATA_PATH = "/root/workspace/evolux/output/mlebench/tabular-playground-series-dec-2021/prepared/public"
OUTPUT_DATA_PATH = "output/ead961ce-50a1-41ec-89e9-4cc0d527fbe5/1/executor/output"

# For type hinting, DT is assumed to be pandas DataFrame or Series
DT = pd.DataFrame | pd.Series


def load_data(validation_mode: bool = False) -> Tuple[DT, DT, DT, DT]:
    """
    Loads, splits, and returns the initial datasets using GPU acceleration.
    """
    train_path = os.path.join(BASE_DATA_PATH, "train.csv")
    test_path = os.path.join(BASE_DATA_PATH, "test.csv")

    # Determine number of rows to load
    nrows = 50 if validation_mode else None

    # Load data using cuDF for GPU-accelerated reading
    train_df = cudf.read_csv(train_path, nrows=nrows)
    test_df = cudf.read_csv(test_path, nrows=nrows)

    # Cleaning: Remove the single sample where Cover_Type == 5
    # (Identified in EDA as having only 1 occurrence in the 4M dataset)
    train_df = train_df[train_df['Cover_Type'] != 5]

    # Extract identifiers and labels before column dropping
    test_ids = test_df['Id'].astype('int32')
    y = train_df['Cover_Type']

    # Cleaning: Drop Id and constant features (Soil_Type7 and Soil_Type15 have std=0)
    cols_to_drop = ['Id', 'Soil_Type7', 'Soil_Type15']
    X = train_df.drop(columns=cols_to_drop + ['Cover_Type'])
    X_test = test_df.drop(columns=cols_to_drop)

    # Memory Optimization: Cast float64 -> float32 and int64 -> int32
    for col in X.columns:
        if X[col].dtype == 'float64':
            X[col] = X[col].astype('float32')
            X_test[col] = X_test[col].astype('float32')
        elif X[col].dtype == 'int64':
            X[col] = X[col].astype('int32')
            X_test[col] = X_test[col].astype('int32')

    # Label Encoding: Map Cover_Type 1-7 (excluding 5) to range [0, 5] for LightGBM
    # Mapping: 1->0, 2->1, 3->2, 4->3, 6->4, 7->5
    mapping = {1: 0, 2: 1, 3: 2, 4: 3, 6: 4, 7: 5}
    y = y.map(mapping).astype('int32')

    # Return as pandas objects as per function specification
    return X.to_pandas(), y.to_pandas(), X_test.to_pandas(), test_ids.to_pandas()
