import os
import cudf
import numpy as np
import cupy as cp
from typing import Tuple, Any, Dict
from cuml.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer
from scipy import sparse

# Task-adaptive type definitions
# X is a sparse matrix containing both GBDT and Transformer features to satisfy length checks
X = sparse.csr_matrix
y = np.ndarray

def preprocess(
    X_train: cudf.DataFrame,
    y_train: cudf.Series,
    X_val: cudf.DataFrame,
    y_val: cudf.Series,
    X_test: cudf.DataFrame
) -> Tuple[X, y, X, y, X]:
    """
    Transforms raw essay data into a unified model-ready format using a dual-stream approach.
    
    Streams:
    1. GBDT: TF-IDF (Word/Char) + Structural Linguistic Features.
    2. Transformer: Tokenized IDs and Attention Masks (DeBERTa-v3).
    
    The outputs are concatenated into a single sparse CSR matrix to ensure row alignment 
    and compatibility with common pipeline validation checks (len(X) == len(y)).
    """
    print(f"Stage 3: Preprocessing starting. Samples - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # Helper: Extract structural features on GPU
    def get_structural_features(df: cudf.DataFrame) -> np.ndarray:
        text_col = df['full_text']
        word_count = text_col.str.split().list.len().astype('float32')
        char_count = text_col.str.len().astype('float32')
        sentence_count = text_col.str.count(r'[.!?]+').astype('float32') + 1.0
        avg_word_len = char_count / (word_count + 1e-5)
        punct_count = text_col.str.count(r'[.,!?;:]').astype('float32')
        punct_density = punct_count / (char_count + 1e-5)
        
        struct_df = cudf.DataFrame({
            'wc': word_count, 'cc': char_count, 'sc': sentence_count,
            'awl': avg_word_len, 'pd': punct_density
        })
        return np.nan_to_num(struct_df.to_pandas().values.astype('float32'), nan=0.0)

    # --- 1. GBDT Stream: Linguistic & TF-IDF Features ---
    print("Computing structural features...")
    struct_train = get_structural_features(X_train)
    struct_val = get_structural_features(X_val)
    struct_test = get_structural_features(X_test)

    # Note: 'strip_accents' is not supported in cuML TfidfVectorizer; 
    # we omit it to maintain GPU acceleration as per hardware context.
    tfidf_word = TfidfVectorizer(ngram_range=(1, 3), max_features=10000, sublinear_tf=True, analyzer='word')
    tfidf_char = TfidfVectorizer(ngram_range=(3, 6), max_features=10000, sublinear_tf=True, analyzer='char')

    print("Fitting TF-IDF Vectorizers (Word & Char) on GPU...")
    train_word = tfidf_word.fit_transform(X_train['full_text']).get()
    val_word = tfidf_word.transform(X_val['full_text']).get()
    test_word = tfidf_word.transform(X_test['full_text']).get()

    train_char = tfidf_char.fit_transform(X_train['full_text']).get()
    val_char = tfidf_char.transform(X_val['full_text']).get()
    test_char = tfidf_char.transform(X_test['full_text']).get()

    # Combine GBDT components
    def assemble_gbdt(word_sp, char_sp, struct_np):
        return sparse.hstack([word_sp, char_sp, sparse.csr_matrix(struct_np)]).tocsr()

    X_train_gbdt = assemble_gbdt(train_word, train_char, struct_train)
    X_val_gbdt = assemble_gbdt(val_word, val_char, struct_val)
    X_test_gbdt = assemble_gbdt(test_word, test_char, struct_test)

    # --- 2. Transformer Stream: Tokenization ---
    print("Tokenizing with DeBERTa-v3-base...")
    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")

    def tokenize_texts(df: cudf.DataFrame) -> Dict[str, np.ndarray]:
        texts = df['full_text'].to_pandas().tolist()
        tokens = tokenizer.batch_encode_plus(
            texts, max_length=1024, padding='max_length', truncation=True, return_tensors='np'
        )
        return {
            'ids': tokens['input_ids'].astype(np.float32), 
            'mask': tokens['attention_mask'].astype(np.float32)
        }

    train_trans = tokenize_texts(X_train)
    val_trans = tokenize_texts(X_val)
    test_trans = tokenize_texts(X_test)

    # --- 3. Final Assembly ---
    # We concatenate GBDT and Transformer features into a single sparse matrix.
    # This ensures len(X_processed) == len(y_processed), avoiding pipeline validation errors.
    def finalize_features(gbdt_sp, trans_dict):
        return sparse.hstack([
            gbdt_sp, 
            sparse.csr_matrix(trans_dict['ids']), 
            sparse.csr_matrix(trans_dict['mask'])
        ]).tocsr()

    X_train_processed = finalize_features(X_train_gbdt, train_trans)
    X_val_processed = finalize_features(X_val_gbdt, val_trans)
    X_test_processed = finalize_features(X_test_gbdt, test_trans)

    y_train_processed = y_train.to_pandas().values.astype(np.int64)
    y_val_processed = y_val.to_pandas().values.astype(np.int64)

    # Alignment check
    if X_train_processed.shape[0] != len(y_train_processed):
        raise ValueError(f"Alignment Error: X_train has {X_train_processed.shape[0]} rows, y_train has {len(y_train_processed)} rows.")
    if X_val_processed.shape[0] != len(y_val_processed):
        raise ValueError(f"Alignment Error: X_val has {X_val_processed.shape[0]} rows, y_val has {len(y_val_processed)} rows.")

    print(f"Preprocessing complete. Feature count: {X_train_processed.shape[1]}")
    return X_train_processed, y_train_processed, X_val_processed, y_val_processed, X_test_processed