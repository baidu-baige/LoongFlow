import numpy as np
import pandas as pd
import os
from typing import Tuple, Dict, Any
from transformers import AutoTokenizer

# Set environment variable to avoid potential hangs with tokenizer parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Define a custom dictionary-like container that reports the number of samples as its length.
# This is necessary because the ML pipeline checks len(X) == len(y), and for a standard 
# dict of arrays, len(X) would return the number of keys (e.g., 2 for 'input_ids' and 'attention_mask').
class ModelInputs(dict):
    """
    A dictionary wrapper for model inputs ('input_ids', 'attention_mask') 
    that returns the sample count for len() to pass pipeline validation.
    """
    def __len__(self) -> int:
        # Check if the dictionary is empty using the base dict class method to avoid recursion
        if dict.__len__(self) == 0:
            return 0
        # Return the length of the first array in the dictionary (number of samples)
        return len(next(iter(self.values())))

# Task-adaptive type definitions
X = ModelInputs
y = np.ndarray

def preprocess(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_val: pd.DataFrame,
    X_test: pd.DataFrame
) -> Tuple[X, y, X, y, X]:
    """
    Transforms raw comment text into tokenized sequences compatible with DeBERTa-v3.

    Args:
        X_train (pd.DataFrame): Training features containing 'comment_text'.
        y_train (pd.DataFrame): Training targets (multi-label binary).
        X_val (pd.DataFrame): Validation features containing 'comment_text'.
        y_val (pd.DataFrame): Validation targets (multi-label binary).
        X_test (pd.DataFrame): Test features containing 'comment_text'.

    Returns:
        Tuple[X, y, X, y, X]: Transformed model-ready data.
    """
    print(f"Preprocessing starting. Samples: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")

    # Load the tokenizer for the specified model
    model_name = "microsoft/deberta-v3-base"
    print(f"Loading tokenizer: {model_name}")
    # DeBERTa-v3 uses SentencePiece; use_fast=False is standard for stability with this architecture
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    
    # Technical Specification: 512 tokens covers most content while fitting memory constraints
    MAX_LENGTH = 512

    def encode_batch(df: pd.DataFrame) -> ModelInputs:
        """Tokenizes a DataFrame's 'comment_text' column into a ModelInputs object."""
        if df.empty:
            return ModelInputs({
                'input_ids': np.empty((0, MAX_LENGTH), dtype=np.int32),
                'attention_mask': np.empty((0, MAX_LENGTH), dtype=np.int32)
            })
            
        # Ensure text is cleaned and handle any missing values
        texts = df['comment_text'].fillna("").astype(str).tolist()
        
        # Tokenize using HuggingFace tokenizer
        encodings = tokenizer(
            texts,
            max_length=MAX_LENGTH,
            padding='max_length',
            truncation=True,
            add_special_tokens=True,
            return_tensors='np'
        )
        
        # Store as ModelInputs to satisfy pipeline length requirements
        return ModelInputs({
            'input_ids': encodings['input_ids'].astype(np.int32),
            'attention_mask': encodings['attention_mask'].astype(np.int8)
        })

    # Transform features for all sets
    print("Tokenizing training set...")
    X_train_processed = encode_batch(X_train)
    
    print("Tokenizing validation set...")
    X_val_processed = encode_batch(X_val)
    
    print("Tokenizing test set...")
    X_test_processed = encode_batch(X_test)

    # Transform targets to float32 numpy arrays for multi-label classification
    y_train_processed = y_train.values.astype(np.float32)
    y_val_processed = y_val.values.astype(np.float32)

    # Sanity check for row alignment and completeness
    if len(X_train_processed) != len(y_train_processed):
        raise ValueError(f"Alignment Error: X_train ({len(X_train_processed)}) vs y_train ({len(y_train_processed)})")
    if len(X_val_processed) != len(y_val_processed):
        raise ValueError(f"Alignment Error: X_val ({len(X_val_processed)}) vs y_val ({len(y_val_processed)})")
    if len(X_test_processed) != len(X_test):
        raise ValueError(f"Completeness Error: X_test ({len(X_test_processed)}) vs Input ({len(X_test)})")

    # Ensure no NaN or Inf values in targets
    if np.isnan(y_train_processed).any() or np.isinf(y_train_processed).any():
        raise ValueError("Training targets contain NaN or Infinity values.")

    print("Preprocessing completed successfully.")
    
    return (
        X_train_processed,
        y_train_processed,
        X_val_processed,
        y_val_processed,
        X_test_processed
    )