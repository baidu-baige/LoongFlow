import pandas as pd
import numpy as np
import re
import os
from typing import Tuple, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer

# Set environment variable to handle tokenizer parallelism warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Use numpy array as the feature matrix type to ensure len(X) == n_samples
X_type = np.ndarray
y_type = pd.Series


def preprocess(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame
) -> Tuple[X_type, y_type, X_type, y_type, X_type]:
    """
    Transforms raw data into model-ready format using dual-path preprocessing:
    Path A: Transformer-based tokenization (DeBERTa-v3-base).
    Path B: Traditional TF-IDF and temporal feature engineering.

    All features are concatenated into a single dense numpy array to ensure
    compatibility with the pipeline's structure validation.
    """
    print("Stage 3: Starting preprocessing...")

    # --- Path B: Traditional Feature Engineering ---

    def clean_text(text: Any) -> str:
        """Cleans text as per specification: remove HTML, lowercase."""
        if not isinstance(text, str):
            text = str(text)
        # Lowercase
        text = text.lower()
        # Remove HTML tags using regex
        text = re.sub(r'<[^>]*>', ' ', text)
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def extract_date_features(df: pd.DataFrame) -> np.ndarray:
        """Extracts Hour, DayOfWeek, and IsMissing features from the 'Date' column."""
        # Ensure 'Date' is datetime (load_data should have handled this)
        dates = pd.to_datetime(df['Date'])
        # Hour: 0-23, fill missing with -1
        hour = dates.dt.hour.fillna(-1).values
        # DayOfWeek: 0-6, fill missing with -1
        dayofweek = dates.dt.dayofweek.fillna(-1).values
        # IsMissing: 1 if missing, 0 otherwise
        is_missing = dates.isna().astype(np.float32).values
        return np.column_stack([hour, dayofweek, is_missing]).astype(np.float32)

    print("Executing Path B: Cleaning text and extracting temporal features...")
    # Apply cleaning to all sets for TF-IDF
    train_comments_clean = X_train['Comment'].apply(clean_text)
    val_comments_clean = X_val['Comment'].apply(clean_text)
    test_comments_clean = X_test['Comment'].apply(clean_text)

    # Extract temporal features
    train_date_feats = extract_date_features(X_train)
    val_date_feats = extract_date_features(X_val)
    test_date_feats = extract_date_features(X_test)

    # Fit TF-IDF on training set only
    print(f"Fitting TF-IDF (max_features=20000, ngrams=1-3)...")
    tfidf = TfidfVectorizer(
        ngram_range=(1, 3),
        max_features=20000,
        sublinear_tf=True,
        dtype=np.float32
    )

    # We use toarray() to convert sparse to dense since we have 440GB RAM
    # and the pipeline runner expects an object where len(X) is the sample count.
    tfidf_train = tfidf.fit_transform(train_comments_clean).toarray()
    tfidf_val = tfidf.transform(val_comments_clean).toarray()
    tfidf_test = tfidf.transform(test_comments_clean).toarray()

    # --- Path A: Transformer Preprocessing ---

    print("Executing Path A: Tokenizing with DeBERTa-v3-base...")
    # Initialize tokenizer (DeBERTa-v3-base)
    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base", use_fast=True)

    def tokenize_data(comments: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        # Handle potential non-string inputs just in case
        text_list = comments.astype(str).tolist()
        encoded = tokenizer.batch_encode_plus(
            text_list,
            add_special_tokens=True,
            max_length=256,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='np'
        )
        return encoded['input_ids'].astype(np.float32), encoded['attention_mask'].astype(np.float32)

    # Tokenize all sets
    train_ids, train_masks = tokenize_data(X_train['Comment'])
    val_ids, val_masks = tokenize_data(X_val['Comment'])
    test_ids, test_masks = tokenize_data(X_test['Comment'])

    # --- Combine All Features ---

    print("Combining features into final dense arrays...")

    def combine_features(tfidf_feats, date_feats, ids, masks):
        # Concatenate horizontally: [TF-IDF (20000), Dates (3), IDs (256), Masks (256)]
        # Total columns = 20515
        return np.hstack([tfidf_feats, date_feats, ids, masks])

    X_train_processed = combine_features(tfidf_train, train_date_feats, train_ids, train_masks)
    X_val_processed = combine_features(tfidf_val, val_date_feats, val_ids, val_masks)
    X_test_processed = combine_features(tfidf_test, test_date_feats, test_ids, test_masks)

    # Final sanity checks
    # Row alignment
    if len(X_train_processed) != len(y_train):
        raise RuntimeError(f"Alignment Error: X_train ({len(X_train_processed)}) vs y_train ({len(y_train)})")
    if len(X_val_processed) != len(y_val):
        raise RuntimeError(f"Alignment Error: X_val ({len(X_val_processed)}) vs y_val ({len(y_val)})")
    if len(X_test_processed) != len(X_test):
        raise RuntimeError(f"Alignment Error: X_test ({len(X_test_processed)}) vs Input X_test ({len(X_test)})")

    # Column consistency
    if not (X_train_processed.shape[1] == X_val_processed.shape[1] == X_test_processed.shape[1]):
        raise RuntimeError(
            f"Column Consistency Error: Shapes {X_train_processed.shape[1]}, {X_val_processed.shape[1]}, {X_test_processed.shape[1]}")

    # Check for NaNs or Infs
    if np.isnan(X_train_processed).any() or np.isinf(X_train_processed).any():
        # Fill NaNs/Infs with 0 as a safety measure, though they shouldn't occur
        X_train_processed = np.nan_to_num(X_train_processed)
        X_val_processed = np.nan_to_num(X_val_processed)
        X_test_processed = np.nan_to_num(X_test_processed)

    print(f"Preprocessing complete. Total features per sample: {X_train_processed.shape[1]}")

    return (
        X_train_processed,
        y_train,
        X_val_processed,
        y_val,
        X_test_processed
    )