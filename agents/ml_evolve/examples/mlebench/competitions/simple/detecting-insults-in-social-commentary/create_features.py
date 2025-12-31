# -*- coding: utf-8 -*-
"""
LLM Generated Code
"""

import re
from typing import Tuple

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

BASE_DATA_PATH = "/root/workspace/evolux/output/mlebench/detecting-insults-in-social-commentary/prepared/public"
OUTPUT_DATA_PATH = "output/bb8b0571-64fb-49b6-8e49-1fc2d52da49b/2/executor/output"

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
    Creates features for a single fold of cross-validation using Word TF-IDF, 
    Character TF-IDF, and Date-based features.
    """

    def clean_text(text: str) -> str:
        if not isinstance(text, str):
            return ""
        # 1. Decode unicode-escaped characters
        try:
            # Handle cases where literal escape sequences might still exist
            text = text.encode('latin-1').decode('unicode_escape')
        except (UnicodeEncodeError, UnicodeDecodeError):
            pass

        # 2. Convert to lowercase
        text = text.lower()

        # 3. Remove non-alphanumeric but keep ! and *
        # This replaces other punctuation and control characters (like \n) with space
        text = re.sub(r'[^a-z0-9\!\*\s]', ' ', text)

        # Collapse multiple whitespaces and strip
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    # Preprocess comments for all sets
    train_comments = X_train['Comment'].apply(clean_text)
    val_comments = X_val['Comment'].apply(clean_text)
    test_comments = X_test['Comment'].apply(clean_text)

    # Word TF-IDF Vectorization
    # Using a custom token_pattern to ensure '!' and '*' are preserved as tokens
    word_vec = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.9,
        sublinear_tf=True,
        token_pattern=r"(?u)[a-z0-9\!\*]+"
    )
    X_train_word = word_vec.fit_transform(train_comments)
    X_val_word = word_vec.transform(val_comments)
    X_test_word = word_vec.transform(test_comments)

    # Character TF-IDF Vectorization
    char_vec = TfidfVectorizer(
        ngram_range=(2, 5),
        min_df=5,
        max_df=0.9,
        analyzer='char_wb',
        sublinear_tf=True
    )
    X_train_char = char_vec.fit_transform(train_comments)
    X_val_char = char_vec.transform(val_comments)
    X_test_char = char_vec.transform(test_comments)

    # Date Feature Extraction
    def extract_date_features(df: pd.DataFrame) -> np.ndarray:
        # Assumes df['Date'] is already converted to datetime/NaT by load_data
        hour = df['Date'].dt.hour.fillna(-1)
        dow = df['Date'].dt.dayofweek.fillna(-1)
        is_missing = df['Date'].isna().astype(float)
        return np.column_stack([hour, dow, is_missing])

    date_train = extract_date_features(X_train)
    date_val = extract_date_features(X_val)
    date_test = extract_date_features(X_test)

    # Scale Date Features
    scaler = StandardScaler()
    date_train_scaled = scaler.fit_transform(date_train)
    date_val_scaled = scaler.transform(date_val)
    date_test_scaled = scaler.transform(date_test)

    # Combine all features (Sparse Word + Sparse Char + Dense Date)
    X_train_combined = hstack([X_train_word, X_train_char, date_train_scaled])
    X_val_combined = hstack([X_val_word, X_val_char, date_val_scaled])
    X_test_combined = hstack([X_test_word, X_test_char, date_test_scaled])

    # Construct feature names for column consistency and prefix them to avoid collisions
    feature_names = (
            ["w_" + f for f in word_vec.get_feature_names_out()] +
            ["c_" + f for f in char_vec.get_feature_names_out()] +
            ["date_hour", "date_dow", "date_is_missing"]
    )

    # Convert combined sparse matrices back to DataFrames to match DT type hint
    # Dense conversion is safe given the hardware context (468GB RAM) and dataset size
    X_train_transformed = pd.DataFrame(
        X_train_combined.toarray(),
        columns=feature_names,
        index=X_train.index
    )
    X_val_transformed = pd.DataFrame(
        X_val_combined.toarray(),
        columns=feature_names,
        index=X_val.index
    )
    X_test_transformed = pd.DataFrame(
        X_test_combined.toarray(),
        columns=feature_names,
        index=X_test.index
    )

    # Ensure no NaN or Infinity values (replace any accidental NaNs with 0)
    X_train_transformed = X_train_transformed.fillna(0).replace([np.inf, -np.inf], 0)
    X_val_transformed = X_val_transformed.fillna(0).replace([np.inf, -np.inf], 0)
    X_test_transformed = X_test_transformed.fillna(0).replace([np.inf, -np.inf], 0)

    # Labels remain unchanged
    y_train_transformed = y_train
    y_val_transformed = y_val

    return (
        X_train_transformed,
        y_train_transformed,
        X_val_transformed,
        y_val_transformed,
        X_test_transformed
    )
