# -*- coding: utf-8 -*-
"""
LLM Generated Code
"""

import re
from typing import Tuple

import numpy as np
import pandas as pd

BASE_DATA_PATH = "/root/workspace/evolux/output/mlebench/text-normalization-challenge-russian-language/prepared/public"
OUTPUT_DATA_PATH = "output/dce1a922-fb4b-4006-9d03-6f53b7ea0718/1/executor/output"

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

    # Define regex patterns
    CYRILLIC_PATTERN = re.compile(r'[а-яА-ЯёЁ]')
    LATIN_PATTERN = re.compile(r'[a-zA-Z]')
    NUMERIC_PATTERN = re.compile(r'\d')
    DATE_PATTERN = re.compile(
        r'\d{1,2}\s+(января|февраля|марта|апреля|мая|июня|июля|августа|сентября|октября|ноября|декабря|\d{1,2})\s*\d{0,4}|\d{4}\s*(год|года|году)',
        re.IGNORECASE)
    CURRENCY_PATTERN = re.compile(r'(рубл|доллар|евро|цент|\$|€|₽|USD|EUR|RUB)', re.IGNORECASE)
    MEASURE_PATTERN = re.compile(
        r'(км|кг|м|см|мм|г|мг|л|мл|га|т|ч|мин|сек|с|°|градус|метр|килограмм|грамм|литр|миллилитр|километр|сантиметр|миллиметр|тонн|час|минут|секунд)',
        re.IGNORECASE)
    TIME_PATTERN = re.compile(r'\d{1,2}:\d{2}(:\d{2})?')
    ROMAN_NUMERAL_PATTERN = re.compile(r'^[IVXLCDM]+$', re.IGNORECASE)
    PUNCTUATION_SET = set('.,;:!?()[]{}"\'-—–«»…')

    def extract_token_features(df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
        """Extract features from a dataframe containing tokens."""

        # Make a copy to avoid modifying original
        result = df.copy()

        # Ensure 'before' column is string type and handle NaN
        result['before'] = result['before'].fillna('').astype(str)

        # Token-level features
        result['is_numeric'] = result['before'].apply(lambda x: bool(NUMERIC_PATTERN.search(x)) if x else False).astype(
            int)
        result['is_all_caps'] = result['before'].apply(lambda x: x.isupper() and len(x) > 0 if x else False).astype(int)
        result['is_punctuation'] = result['before'].apply(
            lambda x: all(c in PUNCTUATION_SET for c in x) and len(x) > 0 if x else False).astype(int)
        result['has_cyrillic'] = result['before'].apply(
            lambda x: bool(CYRILLIC_PATTERN.search(x)) if x else False).astype(int)
        result['has_latin'] = result['before'].apply(lambda x: bool(LATIN_PATTERN.search(x)) if x else False).astype(
            int)
        result['token_length'] = result['before'].apply(lambda x: len(x) if x else 0).astype(int)

        # Digit ratio
        def calc_digit_ratio(x):
            if not x or len(x) == 0:
                return 0.0
            digit_count = sum(1 for c in x if c.isdigit())
            return digit_count / len(x)

        result['digit_ratio'] = result['before'].apply(calc_digit_ratio).astype(float)

        # Pattern-based features
        result['contains_date_pattern'] = result['before'].apply(
            lambda x: bool(DATE_PATTERN.search(x)) if x else False).astype(int)
        result['contains_currency'] = result['before'].apply(
            lambda x: bool(CURRENCY_PATTERN.search(x)) if x else False).astype(int)
        result['contains_measure'] = result['before'].apply(
            lambda x: bool(MEASURE_PATTERN.search(x)) if x else False).astype(int)
        result['contains_time_pattern'] = result['before'].apply(
            lambda x: bool(TIME_PATTERN.search(x)) if x else False).astype(int)
        result['is_roman_numeral'] = result['before'].apply(
            lambda x: bool(ROMAN_NUMERAL_PATTERN.match(x)) if x else False).astype(int)

        # Additional useful features
        result['starts_with_digit'] = result['before'].apply(
            lambda x: x[0].isdigit() if x and len(x) > 0 else False).astype(int)
        result['ends_with_digit'] = result['before'].apply(
            lambda x: x[-1].isdigit() if x and len(x) > 0 else False).astype(int)
        result['is_single_char'] = (result['token_length'] == 1).astype(int)
        result['has_hyphen'] = result['before'].apply(lambda x: '-' in x if x else False).astype(int)
        result['has_space'] = result['before'].apply(lambda x: ' ' in x if x else False).astype(int)

        return result

    def add_context_features(df: pd.DataFrame) -> pd.DataFrame:
        """Add context features based on surrounding tokens."""

        result = df.copy()

        # Sort by sentence_id and token_id to ensure correct order
        result = result.sort_values(['sentence_id', 'token_id']).reset_index(drop=True)

        # Create previous and next token features within each sentence
        result['prev_token'] = result.groupby('sentence_id')['before'].shift(1).fillna('<START>')
        result['next_token'] = result.groupby('sentence_id')['before'].shift(-1).fillna('<END>')

        # Position in sentence
        max_token_per_sentence = result.groupby('sentence_id')['token_id'].transform('max')
        result['position_in_sentence'] = result['token_id'] / (max_token_per_sentence + 1)
        result['position_in_sentence'] = result['position_in_sentence'].fillna(0.0)

        # Is first/last token in sentence
        result['is_first_token'] = (result['token_id'] == 0).astype(int)
        result['is_last_token'] = (result['token_id'] == max_token_per_sentence).astype(int)

        # Features from context tokens
        result['prev_is_numeric'] = result['prev_token'].apply(
            lambda x: bool(NUMERIC_PATTERN.search(x)) if x and x != '<START>' else False).astype(int)
        result['next_is_numeric'] = result['next_token'].apply(
            lambda x: bool(NUMERIC_PATTERN.search(x)) if x and x != '<END>' else False).astype(int)
        result['prev_is_punct'] = result['prev_token'].apply(
            lambda x: all(c in PUNCTUATION_SET for c in x) and len(x) > 0 if x and x != '<START>' else False).astype(
            int)
        result['next_is_punct'] = result['next_token'].apply(
            lambda x: all(c in PUNCTUATION_SET for c in x) and len(x) > 0 if x and x != '<END>' else False).astype(int)

        return result

    # Step 1: Extract token-level features for train and test
    print("Extracting token-level features for training data...")
    X_train_features = extract_token_features(X_train, is_train=True)

    print("Extracting token-level features for test data...")
    X_test_features = extract_token_features(X_test, is_train=False)

    # Step 2: Build lookup features from training data
    print("Building lookup features from training data...")

    # Create mapping from 'before' token to most common class
    train_with_labels = X_train_features.copy()
    train_with_labels['after'] = y_train.values

    # Get vocabulary statistics from training
    before_class_counts = train_with_labels.groupby(['before', 'class']).size().reset_index(name='count')
    idx = before_class_counts.groupby('before')['count'].idxmax()
    most_common_class_map = before_class_counts.loc[idx].set_index('before')['class'].to_dict()

    # Set of tokens seen in training
    train_tokens = set(X_train_features['before'].unique())

    # Add lookup features
    X_train_features['seen_in_train'] = 1  # All training tokens are seen
    X_test_features['seen_in_train'] = X_test_features['before'].apply(lambda x: 1 if x in train_tokens else 0)

    # Map most common class (encode as numeric)
    class_to_idx = {cls: idx for idx, cls in enumerate(sorted(train_with_labels['class'].unique()))}

    def get_class_idx(token, class_map, class_to_idx):
        if token in class_map:
            return class_to_idx.get(class_map[token], -1)
        return -1

    X_train_features['most_common_class_idx'] = X_train_features['before'].apply(
        lambda x: get_class_idx(x, most_common_class_map, class_to_idx)
    )
    X_test_features['most_common_class_idx'] = X_test_features['before'].apply(
        lambda x: get_class_idx(x, most_common_class_map, class_to_idx)
    )

    # Step 3: Add context features
    print("Adding context features for training data...")
    X_train_features = add_context_features(X_train_features)

    print("Adding context features for test data...")
    X_test_features = add_context_features(X_test_features)

    # Step 4: Encode categorical features
    print("Encoding categorical features...")

    # Encode 'class' column for training (test has 'UNKNOWN' placeholder)
    X_train_features['class_encoded'] = X_train_features['class'].map(class_to_idx).fillna(-1).astype(int)
    X_test_features['class_encoded'] = X_test_features['class'].map(class_to_idx).fillna(-1).astype(int)

    # Hash encoding for prev_token and next_token (to handle unseen tokens)
    def hash_token(token, n_buckets=1000):
        if token in ['<START>', '<END>']:
            return hash(token) % n_buckets
        return hash(token) % n_buckets

    X_train_features['prev_token_hash'] = X_train_features['prev_token'].apply(lambda x: hash_token(x))
    X_train_features['next_token_hash'] = X_train_features['next_token'].apply(lambda x: hash_token(x))
    X_test_features['prev_token_hash'] = X_test_features['prev_token'].apply(lambda x: hash_token(x))
    X_test_features['next_token_hash'] = X_test_features['next_token'].apply(lambda x: hash_token(x))

    # Step 5: Select final numeric features
    numeric_features = [
        'is_numeric',
        'is_all_caps',
        'is_punctuation',
        'has_cyrillic',
        'has_latin',
        'token_length',
        'digit_ratio',
        'contains_date_pattern',
        'contains_currency',
        'contains_measure',
        'contains_time_pattern',
        'is_roman_numeral',
        'starts_with_digit',
        'ends_with_digit',
        'is_single_char',
        'has_hyphen',
        'has_space',
        'position_in_sentence',
        'is_first_token',
        'is_last_token',
        'prev_is_numeric',
        'next_is_numeric',
        'prev_is_punct',
        'next_is_punct',
        'seen_in_train',
        'most_common_class_idx',
        'class_encoded',
        'prev_token_hash',
        'next_token_hash'
    ]

    # Also keep original columns needed for prediction
    keep_columns = ['sentence_id', 'token_id', 'before', 'id']

    # Create final feature dataframes
    X_train_transformed = X_train_features[keep_columns + numeric_features].copy()
    X_test_transformed = X_test_features[keep_columns + numeric_features].copy()

    # Ensure no NaN or Infinity values in numeric columns
    for col in numeric_features:
        X_train_transformed[col] = X_train_transformed[col].fillna(0)
        X_test_transformed[col] = X_test_transformed[col].fillna(0)

        # Replace infinity values
        X_train_transformed[col] = X_train_transformed[col].replace([np.inf, -np.inf], 0)
        X_test_transformed[col] = X_test_transformed[col].replace([np.inf, -np.inf], 0)

    # y_train remains unchanged (it's the target text)
    y_train_transformed = y_train.copy()

    # Verify requirements
    assert len(X_test_transformed) == len(
        X_test), f"Test row count mismatch: {len(X_test_transformed)} vs {len(X_test)}"
    assert len(X_train_transformed) == len(
        y_train_transformed), f"Train alignment mismatch: {len(X_train_transformed)} vs {len(y_train_transformed)}"
    assert X_train_transformed.shape[1] == X_test_transformed.shape[
        1], f"Column count mismatch: {X_train_transformed.shape[1]} vs {X_test_transformed.shape[1]}"

    # Check for NaN/Inf in numeric columns
    for col in numeric_features:
        assert not X_train_transformed[col].isna().any(), f"NaN found in train column {col}"
        assert not X_test_transformed[col].isna().any(), f"NaN found in test column {col}"
        assert not np.isinf(X_train_transformed[col]).any(), f"Inf found in train column {col}"
        assert not np.isinf(X_test_transformed[col]).any(), f"Inf found in test column {col}"

    print(f"\nFeature engineering complete:")
    print(f"  X_train_transformed shape: {X_train_transformed.shape}")
    print(f"  y_train_transformed shape: {y_train_transformed.shape}")
    print(f"  X_test_transformed shape: {X_test_transformed.shape}")
    print(f"  Numeric features: {len(numeric_features)}")

    return X_train_transformed, y_train_transformed, X_test_transformed
