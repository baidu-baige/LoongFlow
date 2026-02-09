import pandas as pd
import numpy as np
import scipy.sparse as sp
import re
from typing import Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from collections import Counter

# Task-adaptive type definitions
# Features are represented as sparse TF-IDF matrices (float32 for memory efficiency)
# Targets are represented as sparse binary matrices (indicator format for multi-label)
X = sp.csr_matrix
y = sp.csr_matrix

def preprocess(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame
) -> Tuple[X, y, X, y, X]:
    """
    Transforms raw Stack Exchange question data into model-ready sparse features.
    
    Processing steps:
    1. HTML tag removal from Body content.
    2. Text concatenation of Title (weighted 3x) and Body.
    3. TF-IDF vectorization with n-grams (1-3) and 200,000 features.
    4. Multi-label encoding of the top 5000 tags using a sparse binary matrix.
    """
    
    def prepare_text_features(df: pd.DataFrame) -> pd.Series:
        """Helper to clean HTML and concatenate weighted Title with Body."""
        # Step 1: HTML cleaning using regex
        # We replace tags with a space to ensure word boundaries are preserved
        body_clean = df['Body'].fillna('').str.replace(r'<[^>]*>', ' ', regex=True)
        
        # Step 2: weighted Title concatenation
        # The specification requires Title * 3 emphasis. 
        # We add a trailing space to Title to ensure words are not merged when repeated.
        title_weighted = (df['Title'].fillna('') + " ") * 3
        
        # Combine weighted title and cleaned body
        return title_weighted + body_clean

    print("Stage 1/4: Executing text cleaning and concatenation...")
    X_train_text = prepare_text_features(X_train)
    X_val_text = prepare_text_features(X_val)
    X_test_text = prepare_text_features(X_test)

    print(f"Stage 2/4: Vectorizing text into TF-IDF (max_features=200000, ngrams=1-3)...")
    # Initialize TF-IDF with parameters optimized for keyword extraction
    # sublinear_tf=True applies 1 + log(tf) scaling to dampen the influence of very frequent words
    tfidf = TfidfVectorizer(
        ngram_range=(1, 3),
        max_features=200000,
        stop_words='english',
        sublinear_tf=True,
        dtype=np.float32
    )
    
    # Fit vectorizer on training data and transform all splits
    X_train_processed = tfidf.fit_transform(X_train_text)
    X_val_processed = tfidf.transform(X_val_text)
    X_test_processed = tfidf.transform(X_test_text)

    print("Stage 3/4: Encoding target labels (Top 5000 tags)...")
    # Tags are provided as space-delimited strings
    y_train_split = y_train.fillna('').str.split()
    y_val_split = y_val.fillna('').str.split()

    # Identify top 5000 tags to keep label space manageable and cover ~90% of instances
    tag_counts = Counter()
    for tags in y_train_split:
        tag_counts.update(tags)
    
    top_5000_tags = [tag for tag, count in tag_counts.most_common(5000)]
    
    # Use MultiLabelBinarizer with sparse output to handle high-cardinality multi-label space
    mlb = MultiLabelBinarizer(classes=top_5000_tags, sparse_output=True)
    
    # Transform labels; tags outside the top 5000 will be ignored
    y_train_processed = mlb.fit_transform(y_train_split)
    y_val_processed = mlb.transform(y_val_split)

    print("Stage 4/4: Final validation of output structures...")
    # Ensure row counts match between features and targets
    assert X_train_processed.shape[0] == y_train_processed.shape[0], "Train row mismatch"
    assert X_val_processed.shape[0] == y_val_processed.shape[0], "Val row mismatch"
    assert X_test_processed.shape[0] == len(X_test), "Test completeness failure"
    
    # Ensure column consistency across feature sets
    assert X_train_processed.shape[1] == X_val_processed.shape[1] == X_test_processed.shape[1], "Feature dimension mismatch"

    # Verify absence of NaN/Inf in the transformation results
    for name, mat in [("X_train", X_train_processed), ("X_val", X_val_processed), ("X_test", X_test_processed)]:
        if not np.all(np.isfinite(mat.data)):
            raise ValueError(f"Transformation produced non-finite values in {name}")

    print(f"Preprocessing complete. Sparse feature matrix shape: {X_train_processed.shape}")
    print(f"Target indicator matrix shape: {y_train_processed.shape}")

    return X_train_processed, y_train_processed, X_val_processed, y_val_processed, X_test_processed