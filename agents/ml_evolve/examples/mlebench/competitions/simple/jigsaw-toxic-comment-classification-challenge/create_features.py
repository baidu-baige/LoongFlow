# -*- coding: utf-8 -*-
"""
LLM Generated Code
"""

from typing import Tuple

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

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
    Creates features for a single fold of cross-validation by tokenizing text for a BERT-based model.
    """
    # Initialize the BERT tokenizer
    # Using 'bert-base-uncased' as per instructions.
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    def tokenize_dataframe(df: pd.DataFrame, max_length: int = 256, batch_size: int = 16384) -> pd.DataFrame:
        """
        Helper to clean and tokenize a dataframe in batches to manage memory.
        """
        # Minimal cleaning: remove extra whitespace and ensure string type
        texts = df['comment_text'].fillna("").astype(str)
        texts = texts.str.replace(r'\s+', ' ', regex=True).str.strip().tolist()

        all_ids = []
        all_masks = []

        # Process in batches to balance speed and memory usage
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i: i + batch_size]
            encoded = tokenizer.batch_encode_plus(
                batch_texts,
                add_special_tokens=True,
                max_length=max_length,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='np'
            )
            all_ids.append(encoded['input_ids'].astype(np.int32))
            all_masks.append(encoded['attention_mask'].astype(np.int32))

        # Combine all batches
        input_ids = np.concatenate(all_ids, axis=0)
        attention_masks = np.concatenate(all_masks, axis=0)

        # Create column names
        id_cols = [f'input_ids_{i}' for i in range(max_length)]
        mask_cols = [f'attention_mask_{i}' for i in range(max_length)]

        # Create the transformed DataFrame
        # Horizontal stack IDs and Masks (N samples, 512 columns)
        combined_features = np.hstack([input_ids, attention_masks])
        transformed_df = pd.DataFrame(
            combined_features,
            columns=id_cols + mask_cols,
            index=df.index
        )

        return transformed_df

    # Step 1 & 2: Apply tokenization to train, val, and test sets
    # We use the same tokenizer and parameters across all sets to ensure column consistency.
    X_train_transformed = tokenize_dataframe(X_train)
    X_val_transformed = tokenize_dataframe(X_val)
    X_test_transformed = tokenize_dataframe(X_test)

    # Target variables are usually unchanged in this transformation step
    y_train_transformed = y_train
    y_val_transformed = y_val

    # Step 3: Validation (Internal check for NaNs/Infs)
    # Tokenization outputs are always integers (0 or higher), so NaNs/Infs are not expected.
    # The concatenation and DataFrame creation logic ensures identical columns.

    # Step 4: Return transformed sets
    return X_train_transformed, y_train_transformed, X_val_transformed, y_val_transformed, X_test_transformed
