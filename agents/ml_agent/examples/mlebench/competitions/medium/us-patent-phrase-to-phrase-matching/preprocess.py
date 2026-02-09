import pandas as pd
import numpy as np
from transformers import AutoTokenizer
from typing import Tuple, Dict, List

# Task-adaptive type definitions
# X: pd.DataFrame where each row is a sample and columns are 'input_ids', 'attention_mask', etc.
# y: pd.Series containing the target scores
X = pd.DataFrame
y = pd.Series

def preprocess(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame
) -> Tuple[X, y, X, y, X]:
    """
    Transforms raw patent phrase data into tokenized format for DeBERTa-v3-large
    using a structured prompt template.
    """
    print("Execution: preprocess (Stage 3)")
    
    # Step 1: Initialize the tokenizer
    # Using microsoft/deberta-v3-large as per technical specification
    model_name = "microsoft/deberta-v3-large"
    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    
    def tokenize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """
        Concatenates text columns according to the technical specification and tokenizes.
        """
        # Technical Spec Template: 
        # "anchor: " + df['anchor'] + tokenizer.sep_token + "target: " + df['target'] + 
        # tokenizer.sep_token + "context: " + df['context_desc'] + " " + df['context']
        sep = tokenizer.sep_token
        
        print("Constructing structured prompt sequences...")
        texts = (
            "anchor: " + df['anchor'].astype(str) + 
            sep + 
            "target: " + df['target'].astype(str) + 
            sep + 
            "context: " + df['context_desc'].astype(str) + 
            " " + 
            df['context'].astype(str)
        ).tolist()
        
        # Tokenization parameters from technical specification:
        # max_length=172, padding='max_length', truncation=True, return_tensors=None
        print(f"Tokenizing {len(texts)} samples (max_length=172)...")
        encodings = tokenizer(
            texts,
            max_length=172,
            padding='max_length',
            truncation=True,
            return_tensors=None
        )
        
        # Convert dictionary of lists into a DataFrame where each row is a sample
        # Columns: input_ids, attention_mask, and token_type_ids (if provided by tokenizer)
        processed_df = pd.DataFrame({k: list(v) for k, v in encodings.items()})
        
        return processed_df

    # Step 2: Apply transformations to all splits
    print("Processing training set...")
    X_train_processed = tokenize_dataframe(X_train)
    y_train_processed = y_train.reset_index(drop=True)

    print("Processing validation set...")
    X_val_processed = tokenize_dataframe(X_val)
    y_val_processed = y_val.reset_index(drop=True)

    print("Processing test set...")
    X_test_processed = tokenize_dataframe(X_test)

    # Step 3: Consistency and Alignment Checks
    # Row alignment checks
    if len(X_train_processed) != len(y_train_processed):
        raise ValueError(f"Train alignment mismatch: X({len(X_train_processed)}) vs y({len(y_train_processed)})")
    
    if len(X_val_processed) != len(y_val_processed):
        raise ValueError(f"Val alignment mismatch: X({len(X_val_processed)}) vs y({len(y_val_processed)})")
    
    if len(X_test_processed) != len(X_test):
        raise ValueError(f"Test completeness error: X_processed({len(X_test_processed)}) vs X_orig({len(X_test)})")

    # Column consistency check
    if not (X_train_processed.columns.equals(X_val_processed.columns) and 
            X_train_processed.columns.equals(X_test_processed.columns)):
        raise ValueError("Feature columns are inconsistent across splits.")

    # Target integrity check (NaN/Inf)
    if y_train_processed.isna().any() or np.isinf(y_train_processed.values).any():
        raise ValueError("Training target contains NaN or Infinity values.")
    
    if y_val_processed.isna().any() or np.isinf(y_val_processed.values).any():
        raise ValueError("Validation target contains NaN or Infinity values.")

    print(f"Preprocessing complete. Features: {list(X_train_processed.columns)}")
    print(f"Shapes: Train={X_train_processed.shape}, Val={X_val_processed.shape}, Test={X_test_processed.shape}")
    
    return (
        X_train_processed, 
        y_train_processed, 
        X_val_processed, 
        y_val_processed, 
        X_test_processed
    )