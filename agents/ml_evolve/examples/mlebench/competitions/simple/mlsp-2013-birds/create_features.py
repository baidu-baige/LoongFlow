# -*- coding: utf-8 -*-
"""
LLM Generated Code
"""

import os
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

BASE_DATA_PATH = "/root/workspace/evolux/output/mlebench/mlsp-2013-birds/prepared/public"
OUTPUT_DATA_PATH = "output/b4d4dfce-4367-41a9-8cac-a59279a6d65f/11/executor/output"

# For type hinting, DT is assumed to be pandas DataFrame or Series
DT = pd.DataFrame | pd.Series


def create_features(
    X_train: DT,
    y_train: DT,
    X_val: DT,
    y_val: DT,
    X_test: DT
) -> Tuple[DT, DT, DT, DT, DT]:
    """
    Creates features for a single fold of cross-validation.
    
    Strategy:
    1. Loads MIML segment features (`segment_features.txt`).
    2. Aggregates segment features using statistical moments and percentiles (p25, p50, p75).
       - Explicitly excludes skew and kurtosis to reduce noise.
    3. Loads BoW histogram features (`histogram_of_segments.txt`) if available.
    4. Merges these supplementary features with the base features provided in X inputs.
    5. Cleans, aligns, and scales the data.

    Args:
        X_train (DT): The training set features (includes rec_id).
        y_train (DT): The training set labels.
        X_val (DT): The validation set features (includes rec_id).
        y_val (DT): The validation set labels.
        X_test (DT): The test set features (includes rec_id).

    Returns:
        Tuple[DT, DT, DT, DT, DT]: Transformed datasets:
        (X_train_transformed, y_train, X_val_transformed, y_val, X_test_transformed)
    """

    # --- 1. Helper Functions for Data Loading ---

    def load_segment_features() -> pd.DataFrame:
        """Loads and aggregates segment_features.txt."""
        path = os.path.join(BASE_DATA_PATH, "supplemental_data", "segment_features.txt")
        if not os.path.exists(path):
            return pd.DataFrame()

        try:
            # Check if file is empty
            if os.path.getsize(path) == 0:
                return pd.DataFrame()

            # Format: rec_id, segment_id, feat_0 ... feat_37
            # No header in file
            df = pd.read_csv(path, header=None)

            if df.empty or df.shape[1] < 3:
                return pd.DataFrame()

            # Rename columns
            n_feats = df.shape[1] - 2
            feat_cols = [f"seg_{i}" for i in range(n_feats)]
            df.columns = ["rec_id", "segment_id"] + feat_cols

            # Define Aggregation Functions
            def p25(x):
                return x.quantile(0.25)

            def p50(x):
                return x.quantile(0.50)

            def p75(x):
                return x.quantile(0.75)

            # Ensure names are clean for column flattening
            p25.__name__ = 'p25'
            p50.__name__ = 'p50'
            p75.__name__ = 'p75'

            # Aggregation Strategy: Mean, Std, Min, Max, Percentiles
            # Note: Skew and Kurt explicitly excluded
            aggs = ['mean', 'std', 'min', 'max', p25, p50, p75]

            # Group by rec_id and aggregate features
            grouped = df.groupby('rec_id')[feat_cols].agg(aggs)

            # Flatten MultiIndex Columns
            # Output format: seg_0_mean, seg_0_p25, etc.
            new_cols = []
            for feat, stat in grouped.columns:
                stat_name = stat if isinstance(stat, str) else stat.__name__
                new_cols.append(f"{feat}_{stat_name}")

            grouped.columns = new_cols
            return grouped.reset_index()

        except Exception:
            # Graceful fallback
            return pd.DataFrame()

    def load_histogram_features() -> pd.DataFrame:
        """Loads histogram_of_segments.txt."""
        path = os.path.join(BASE_DATA_PATH, "supplemental_data", "histogram_of_segments.txt")
        if not os.path.exists(path):
            return pd.DataFrame()

        try:
            if os.path.getsize(path) == 0:
                return pd.DataFrame()

            df = pd.read_csv(path, header=None)
            if df.empty:
                return pd.DataFrame()

            # Heuristic: Check if first col is rec_id (integer-like)
            # This handles the "bag of words" vector file
            if np.issubdtype(df[0].dtype, np.integer) or (df[0] % 1 == 0).all():
                # Assuming first col is rec_id
                cols = [f"hist_{i}" for i in range(df.shape[1] - 1)]
                df.columns = ["rec_id"] + cols
                return df

            # If structure is unknown, return empty to avoid mismatch
            return pd.DataFrame()
        except Exception:
            return pd.DataFrame()

    # --- 2. Prepare Auxiliary Features ---

    # Load auxiliary data
    df_aggs = load_segment_features()
    df_hist = load_histogram_features()

    # Merge Aux Data sources together first (if both exist)
    aux_features = pd.DataFrame()

    if not df_aggs.empty:
        aux_features = df_aggs

    if not df_hist.empty:
        if aux_features.empty:
            aux_features = df_hist
        else:
            # Outer join to keep information from both sources
            aux_features = pd.merge(aux_features, df_hist, on='rec_id', how='outer')

    # --- 3. Processing Pipeline ---

    scaler = StandardScaler()
    train_columns = []

    def transform_split(X: DT, is_train: bool) -> DT:
        nonlocal train_columns

        df = X.copy()

        # Merge Auxiliary Features
        # Left join ensures we strictly keep the rows defined in the input split (Train/Val/Test)
        if not aux_features.empty and 'rec_id' in df.columns:
            df = pd.merge(df, aux_features, on='rec_id', how='left')

        # Drop ID (not a predictive feature)
        if 'rec_id' in df.columns:
            df = df.drop(columns=['rec_id'])

        # Impute Missing Values (created by left merge where aux data was missing)
        df = df.fillna(0)

        # Handle Infinite Values
        df = df.replace([np.inf, -np.inf], 0)

        # Column Alignment
        if is_train:
            # Define the schema based on training data
            train_columns = df.columns.tolist()
        else:
            # Enforce schema on Val/Test
            if not train_columns:
                # Fallback
                train_columns = df.columns.tolist()

            # Add missing columns (fill with 0)
            missing_cols = set(train_columns) - set(df.columns)
            for c in missing_cols:
                df[c] = 0.0

            # Drop extra columns (not present in train)
            extra_cols = set(df.columns) - set(train_columns)
            if extra_cols:
                df = df.drop(columns=list(extra_cols))

            # Reorder columns to match training exactly
            df = df[train_columns]

        # Scaling
        if is_train:
            arr = scaler.fit_transform(df)
        else:
            arr = scaler.transform(df)

        # Wrap back to DataFrame to preserve metadata if needed (indices)
        df_transformed = pd.DataFrame(arr, columns=df.columns, index=df.index)

        # Final safety fill for any scaling artifacts
        return df_transformed.fillna(0)

    # --- 4. Execution ---

    # Process Training Data (fits scaler)
    X_train_tf = transform_split(X_train, is_train=True)

    # Process Validation and Test Data (transforms using fitted scaler)
    X_val_tf = transform_split(X_val, is_train=False)
    X_test_tf = transform_split(X_test, is_train=False)

    return X_train_tf, y_train, X_val_tf, y_val, X_test_tf
