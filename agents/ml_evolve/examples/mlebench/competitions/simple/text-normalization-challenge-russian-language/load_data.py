# -*- coding: utf-8 -*-
"""
LLM Generated Code
"""

import os
import pickle
from typing import Tuple

import pandas as pd

BASE_DATA_PATH = "/root/workspace/evolux/output/mlebench/text-normalization-challenge-russian-language/prepared/public"
OUTPUT_DATA_PATH = "output/dce1a922-fb4b-4006-9d03-6f53b7ea0718/1/executor/output"

# For type hinting, DT is assumed to be pandas DataFrame or Series
DT = pd.DataFrame | pd.Series


def load_data() -> Tuple[DT, DT, DT, DT]:
    """
    Loads, splits, and returns the initial datasets.

    This function takes no arguments as it should derive file paths from the task description
    or predefined global variables.

    Returns:
        Tuple[DT, DT, DT, DT]: A tuple containing four elements:
        - X (DT): Training data features.
        - y (DT): Training data labels.
        - X_test (DT): Test data features.
        - test_ids (DT): Identifiers for the test data.
    """
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DATA_PATH, exist_ok=True)

    # Define file paths
    train_path = os.path.join(BASE_DATA_PATH, "ru_train.csv.zip")
    test_path = os.path.join(BASE_DATA_PATH, "ru_test_2.csv.zip")
    sample_submission_path = os.path.join(BASE_DATA_PATH, "ru_sample_submission_2.csv.zip")

    # Load training data - using chunked reading due to large size (9.5M rows)
    print("Loading training data...")
    train_chunks = []
    chunk_size = 500000

    for chunk in pd.read_csv(train_path, chunksize=chunk_size):
        train_chunks.append(chunk)

    train_df = pd.concat(train_chunks, ignore_index=True)
    print(f"Training data loaded: {len(train_df)} rows")

    # Load test data
    print("Loading test data...")
    test_df = pd.read_csv(test_path)
    print(f"Test data loaded: {len(test_df)} rows")

    # Load sample submission for reference
    print("Loading sample submission...")
    sample_submission = pd.read_csv(sample_submission_path)
    print(f"Sample submission loaded: {len(sample_submission)} rows")

    # Handle missing values in 'before' column
    train_df['before'] = train_df['before'].fillna('')
    test_df['before'] = test_df['before'].fillna('')

    # Create unique identifier 'id' column
    train_df['id'] = train_df['sentence_id'].astype(str) + '_' + train_df['token_id'].astype(str)
    test_df['id'] = test_df['sentence_id'].astype(str) + '_' + test_df['token_id'].astype(str)

    # Build lookup dictionaries for exact matches (before -> after mapping)
    print("Building lookup dictionaries...")
    before_after_mapping = {}
    class_examples = {}

    # Group by 'before' and find most common 'after' transformation
    before_after_counts = train_df.groupby(['before', 'after']).size().reset_index(name='count')

    # For each 'before' token, get the most common 'after' transformation
    idx = before_after_counts.groupby('before')['count'].idxmax()
    most_common_transformations = before_after_counts.loc[idx]

    for _, row in most_common_transformations.iterrows():
        before_after_mapping[row['before']] = row['after']

    print(f"Built before->after mapping with {len(before_after_mapping)} entries")

    # Build class-specific transformation examples
    for class_name in train_df['class'].unique():
        class_data = train_df[train_df['class'] == class_name]
        # Store sample transformations for each class
        class_examples[class_name] = class_data[['before', 'after']].drop_duplicates().head(1000).to_dict('records')

    print(f"Built class examples for {len(class_examples)} classes")

    # Extract class distribution statistics
    class_distribution = train_df['class'].value_counts().to_dict()
    print(f"Class distribution computed: {len(class_distribution)} classes")

    # Create vocabulary of unique 'before' tokens with their transformations
    vocabulary = {}
    for before_token, group in train_df.groupby('before'):
        after_counts = group['after'].value_counts()
        vocabulary[before_token] = {
            'most_common': after_counts.index[0],
            'count': after_counts.iloc[0],
            'total': len(group),
            'classes': group['class'].value_counts().to_dict()
        }

    print(f"Built vocabulary with {len(vocabulary)} unique tokens")

    # Save auxiliary data structures for later use
    auxiliary_data = {
        'before_after_mapping': before_after_mapping,
        'class_examples': class_examples,
        'class_distribution': class_distribution,
        'vocabulary': vocabulary
    }

    auxiliary_path = os.path.join(OUTPUT_DATA_PATH, 'auxiliary_data.pkl')
    with open(auxiliary_path, 'wb') as f:
        pickle.dump(auxiliary_data, f)
    print(f"Saved auxiliary data to {auxiliary_path}")

    # Prepare features (X) and labels (y) for training
    # Features include: sentence_id, token_id, before, class, id
    feature_columns = ['sentence_id', 'token_id', 'before', 'class', 'id']
    X = train_df[feature_columns].copy()
    y = train_df['after'].copy()

    # Prepare test features (X_test) and test identifiers
    # Note: test set does NOT have 'class' column, so we add a placeholder
    test_df['class'] = 'UNKNOWN'  # Placeholder since test doesn't have class
    X_test = test_df[feature_columns].copy()
    test_ids = test_df['id'].copy()

    # Verify alignment requirements
    assert len(X) == len(y), f"X and y length mismatch: {len(X)} vs {len(y)}"
    assert len(X_test) == len(test_ids), f"X_test and test_ids length mismatch: {len(X_test)} vs {len(test_ids)}"
    assert list(X.columns) == list(X_test.columns), "Column mismatch between X and X_test"

    print(f"\nFinal shapes:")
    print(f"  X: {X.shape}")
    print(f"  y: {y.shape}")
    print(f"  X_test: {X_test.shape}")
    print(f"  test_ids: {test_ids.shape}")

    return X, y, X_test, test_ids
