# -*- coding: utf-8 -*-
"""
LLM Generated Code
"""

import os
from typing import Tuple

import cudf
import pandas as pd

BASE_DATA_PATH = "/root/workspace/evolux_ml/output/mlebench/text-normalization-challenge-english-language/prepared/public"
OUTPUT_DATA_PATH = "output/12e29d80-a70c-426a-9331-de3aa1a6ce7c/14/executor/output"

# For type hinting, DT is assumed to be pandas DataFrame or Series
DT = pd.DataFrame | pd.Series


def load_data(validation_mode: bool = False) -> Tuple[DT, DT, DT, DT]:
    """
    Loads, splits, and returns the initial datasets using GPU acceleration where possible.
    
    Args:
        validation_mode: Controls the data loading behavior.
            - False (default): Load the complete dataset for actual training/inference.
            - True: Load a small subset of data (≤100 rows) for quick code validation.

    Returns:
        Tuple[DT, DT, DT, DT]: (X, y, X_test, test_ids)
    """
    train_path = os.path.join(BASE_DATA_PATH, "en_train.csv.zip")
    test_path = os.path.join(BASE_DATA_PATH, "en_test_2.csv.zip")

    # Determine the number of rows to load
    nrows = 100 if validation_mode else None

    # Load data using pandas first to handle zip decompression and preserve empty tokens
    # keep_default_na=False ensures that tokens like "NA" are not interpreted as nulls
    train_pd = pd.read_csv(train_path, compression='zip', nrows=nrows, keep_default_na=False)
    test_pd = pd.read_csv(test_path, compression='zip', nrows=nrows, keep_default_na=False)

    # Move to GPU (cuDF) for efficient processing
    train_gdf = cudf.from_pandas(train_pd)
    test_gdf = cudf.from_pandas(test_pd)

    # Preprocessing and memory optimization
    # Convert numeric IDs to int32 to save memory
    train_gdf['sentence_id'] = train_gdf['sentence_id'].astype('int32')
    train_gdf['token_id'] = train_gdf['token_id'].astype('int32')
    test_gdf['sentence_id'] = test_gdf['sentence_id'].astype('int32')
    test_gdf['token_id'] = test_gdf['token_id'].astype('int32')

    # Sort by sentence_id and token_id to maintain sequence order
    # This is critical for context window features and sequence-based modeling
    train_gdf = train_gdf.sort_values(['sentence_id', 'token_id']).reset_index(drop=True)
    test_gdf = test_gdf.sort_values(['sentence_id', 'token_id']).reset_index(drop=True)

    # Feature extraction (X and X_test must be identical in columns)
    features = ['sentence_id', 'token_id', 'before']
    X = train_gdf[features]

    # y contains the target 'after' and the 'class' for specialized normalization logic
    y = train_gdf[['after', 'class']]

    X_test = test_gdf[features]

    # Create test identifiers (format: sentence_id_token_id) as required for submission
    test_ids = test_gdf['sentence_id'].astype(str) + "_" + test_gdf['token_id'].astype(str)
    test_ids.name = 'id'

    # Convert back to pandas for the required return type specified in the prompt
    return (
        X.to_pandas(),
        y.to_pandas(),
        X_test.to_pandas(),
        test_ids.to_pandas()
    )
