# -*- coding: utf-8 -*-
"""
LLM Generated Code
"""

from typing import Tuple

import pandas as pd
from transformers import AutoTokenizer

BASE_DATA_PATH = "/root/workspace/evolux/output/mlebench/us-patent-phrase-to-phrase-matching/prepared/public"
OUTPUT_DATA_PATH = "output/aefcc010-8f21-4ecb-b149-7bf99579e6d3/6/executor/output"

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
    Creates features for a single fold of cross-validation using the DeBERTa-v3-large tokenizer.
    
    Args:
        X_train (DT): The training set features.
        y_train (DT): The training set labels.
        X_val (DT): The validation set features.
        y_val (DT): The validation set labels.
        X_test (DT): The test set features.
    
    Returns:
        Tuple[DT, DT, DT, DT, DT]: A tuple containing the transformed data:
            - X_train_transformed (DT): Transformed training features.
            - y_train_transformed (DT): Transformed training labels.
            - X_val_transformed (DT): Transformed validation features.
            - y_val_transformed (DT): Transformed validation labels.
            - X_test_transformed (DT): Transformed test set.
    """
    # Initialize the tokenizer for microsoft/deberta-v3-large
    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-large")

    def transform_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """
        Helper function to apply consistent transformations across datasets.
        """
        # Create a copy to avoid unintended side effects on original dataframes
        df_transformed = df.copy()

        # For column consistency across train, val, and test, remove 'fold' if it exists.
        # 'fold' is used for splitting but is not a model feature.
        if 'fold' in df_transformed.columns:
            df_transformed = df_transformed.drop(columns=['fold'])

        # Step 1: Construct the input sequence as specified:
        # [context_desc] + " [SEP] " + [anchor] + " [SEP] " + [target]
        # This primes the model with domain context before phrase pairs.
        df_transformed['input'] = (
                df_transformed['context_desc'].astype(str) +
                " [SEP] " +
                df_transformed['anchor'].astype(str) +
                " [SEP] " +
                df_transformed['target'].astype(str)
        )

        # Step 2: Tokenize the input sequence
        # We pass the 'input' column as a list to the tokenizer for efficiency.
        # Requirements: max_length=192, padding='max_length', and truncation=True.
        tokenized_output = tokenizer(
            df_transformed['input'].tolist(),
            max_length=192,
            padding='max_length',
            truncation=True,
            add_special_tokens=True
        )

        # Step 3: Extract tokenized features and store them in the dataframe.
        # These are stored as columns containing lists of integers.
        df_transformed['input_ids'] = tokenized_output['input_ids']
        df_transformed['attention_mask'] = tokenized_output['attention_mask']

        return df_transformed

    # Apply the transformation to training, validation, and test datasets
    X_train_transformed = transform_dataframe(X_train)
    X_val_transformed = transform_dataframe(X_val)
    X_test_transformed = transform_dataframe(X_test)

    # Labels remain unchanged as they are already in the correct format (floats)
    y_train_transformed = y_train
    y_val_transformed = y_val

    # Final verification: Check for row preservation and column consistency.
    # All X sets will have columns: [id, anchor, target, context, context_desc, input, input_ids, attention_mask]

    # Ensure no NaN values exist in the features.
    # String concatenation and tokenizer outputs do not produce NaNs.
    if X_train_transformed.isnull().any().any() or X_test_transformed.isnull().any().any():
        raise ValueError("Transformed features contain NaN values.")

    return (
        X_train_transformed,
        y_train_transformed,
        X_val_transformed,
        y_val_transformed,
        X_test_transformed
    )
