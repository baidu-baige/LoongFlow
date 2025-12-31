# -*- coding: utf-8 -*-
"""
LLM Generated Code
"""

import re
from typing import Tuple

import numpy as np
import pandas as pd

BASE_DATA_PATH = "/root/workspace/evolux-ml/output/mlebench/random-acts-of-pizza/prepared/public"
OUTPUT_DATA_PATH = "output/ed330620-ed29-4387-b009-fed5bf45c1a8/11/executor/output"

# For type hinting, DT is assumed to be pandas DataFrame or Series
DT = pd.DataFrame | pd.Series


def create_features(
    X_train: DT,
    y_train: DT,
    X_test: DT
) -> Tuple[DT, DT, DT]:
    """
    Creates features for a single fold of cross-validation.
    
    Args:
        X_train (DT): The training set features (fit encoders here).
        y_train (DT): The training set labels.
        X_test (DT): The test/validation set features (transform only).
    
    Returns:
        Tuple[DT, DT, DT]: A tuple containing the transformed data:
        - X_train_transformed (DT): Transformed training features (numeric).
        - y_train_transformed (DT): Transformed training labels (usually unchanged).
        - X_test_transformed (DT): Transformed test set (numeric).
    """
    # Make copies to avoid modifying original dataframes
    X_train = X_train.copy()
    X_test = X_test.copy()

    # Step 1: Drop highly correlated features (keep only unix_timestamp_of_request_utc)
    timestamp_cols = ['unix_timestamp_of_request', 'unix_timestamp_of_request_utc']
    if all(col in X_train.columns for col in timestamp_cols):
        X_train.drop(columns=['unix_timestamp_of_request'], inplace=True)
        X_test.drop(columns=['unix_timestamp_of_request'], inplace=True)

    # Step 2: Create text-based features (simplified version)
    text_features = ['request_text', 'request_title']

    def clean_text(text):
        if pd.isna(text):
            return ""
        # Remove URLs and multiple spaces
        text = re.sub(r'http\S+|www\S+|https\S+', '', str(text), flags=re.MULTILINE)
        text = re.sub(r'\s+', ' ', text)
        return text.strip().lower()

    # Apply text cleaning and create basic text features
    for col in text_features:
        if col in X_train.columns:
            # Clean text
            X_train[col] = X_train[col].apply(clean_text)
            X_test[col] = X_test[col].apply(clean_text)

            # Create text length and word count features
            X_train[f'{col}_length'] = X_train[col].str.len().fillna(0)
            X_test[f'{col}_length'] = X_test[col].str.len().fillna(0)

            X_train[f'{col}_word_count'] = X_train[col].str.split().str.len().fillna(0)
            X_test[f'{col}_word_count'] = X_test[col].str.split().str.len().fillna(0)

    # Step 3: Create interaction features (vote-related)
    vote_cols = ['number_of_upvotes_of_request_at_retrieval',
                 'number_of_downvotes_of_request_at_retrieval']

    if all(col in X_train.columns for col in vote_cols):
        # Upvote ratio
        total_votes = X_train[vote_cols[0]] + X_train[vote_cols[1]]
        X_train['upvote_ratio'] = np.where(total_votes > 0,
                                           X_train[vote_cols[0]] / total_votes,
                                           0.5)

        total_votes_test = X_test[vote_cols[0]] + X_test[vote_cols[1]]
        X_test['upvote_ratio'] = np.where(total_votes_test > 0,
                                          X_test[vote_cols[0]] / total_votes_test,
                                          0.5)

        # Vote difference
        X_train['vote_difference'] = X_train[vote_cols[0]] - X_train[vote_cols[1]]
        X_test['vote_difference'] = X_test[vote_cols[0]] - X_test[vote_cols[1]]

    # Step 4: Handle requester_user_flair (categorical feature)
    if 'requester_user_flair' in X_train.columns:
        # Fill missing values with 'None' (most common)
        X_train['requester_user_flair'].fillna('None', inplace=True)
        X_test['requester_user_flair'].fillna('None', inplace=True)

        # One-hot encode (only for categories present in train)
        flair_dummies_train = pd.get_dummies(X_train['requester_user_flair'], prefix='flair')
        flair_dummies_test = pd.get_dummies(X_test['requester_user_flair'], prefix='flair')

        # Align test columns with train
        for col in flair_dummies_train.columns:
            if col not in flair_dummies_test.columns:
                flair_dummies_test[col] = 0
        flair_dummies_test = flair_dummies_test[flair_dummies_train.columns]

        # Drop original column and concatenate dummies
        X_train = pd.concat([X_train.drop(columns=['requester_user_flair']), flair_dummies_train], axis=1)
        X_test = pd.concat([X_test.drop(columns=['requester_user_flair']), flair_dummies_test], axis=1)

    # Step 5: Time-based features from unix_timestamp_of_request_utc
    if 'unix_timestamp_of_request_utc' in X_train.columns:
        for df in [X_train, X_test]:
            df['hour_of_day'] = df['unix_timestamp_of_request_utc'].dt.hour
            df['day_of_week'] = df['unix_timestamp_of_request_utc'].dt.dayofweek
            df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

    # Step 6: Drop original text columns if they exist
    for col in text_features:
        if col in X_train.columns:
            X_train.drop(columns=[col], inplace=True)
        if col in X_test.columns:
            X_test.drop(columns=[col], inplace=True)

    # Step 7: Ensure no NaN or infinite values remain
    X_train = X_train.fillna(0)
    X_test = X_test.fillna(0)

    # Replace infinite values with large finite numbers
    X_train = X_train.replace([np.inf, -np.inf], 1e6)
    X_test = X_test.replace([np.inf, -np.inf], 1e6)

    # Step 8: Ensure column consistency between train and test
    # Add missing columns to test with 0 values
    for col in X_train.columns:
        if col not in X_test.columns:
            X_test[col] = 0

    # Remove extra columns from test that aren't in train
    for col in X_test.columns:
        if col not in X_train.columns:
            X_test.drop(columns=[col], inplace=True)

    # Ensure same column order
    X_test = X_test[X_train.columns]

    return X_train, y_train, X_test
