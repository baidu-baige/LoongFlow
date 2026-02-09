import pandas as pd
import numpy as np
import string
import os
import pickle
from transformers import AutoTokenizer
from typing import Tuple, Any

# Task-adaptive type definitions
# Features are returned as a DataFrame where complex vectors are stored as list-objects
X = pd.DataFrame
y = np.ndarray

def preprocess(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame
) -> Tuple[X, y, X, y, X]:
    """
    Transforms raw data into model-ready format for English Text Normalization.
    Outputs include BERT-aligned tokens, character sequences, and a lookup dictionary.
    """
    print("Initializing preprocessing...")
    
    # Parameters from Technical Specification
    max_sent_len = 128
    max_char_len = 32
    OUTPUT_DATA_PATH = "output/9fef8e79-9e97-4657-be88-07dd4ac6f366/1/executor/output"
    
    # Ensure output directory exists
    os.makedirs(OUTPUT_DATA_PATH, exist_ok=True)
    
    # Initialize BERT Tokenizer (Fast version required for word_ids)
    print("Loading DistilBert tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased', use_fast=True)

    # 1. Frequency Dictionary Construction (Fit on Training Data ONLY)
    print("Building frequency lookup dictionary...")
    train_temp = pd.DataFrame({
        'before': X_train['before'].values,
        'class': X_train['class'].values,
        'after': y_train.values
    })
    # Group by (before, class) and select the mode (most frequent 'after' value)
    lookup_map = train_temp.groupby(['before', 'class'], observed=True)['after'].agg(
        lambda x: x.mode().iloc[0] if not x.mode().empty else ""
    ).to_dict()
    
    # Save lookup map for downstream usage (inference/ensemble)
    lookup_path = os.path.join(OUTPUT_DATA_PATH, "lookup_map.pkl")
    with open(lookup_path, 'wb') as f:
        pickle.dump(lookup_map, f)
    del train_temp

    # 2. Character-level Sequence Preparation
    # Vocab size 50: 0=PAD, 1=UNK, 2-49=Common characters
    chars = string.printable[:48] 
    char_to_id = {c: i+2 for i, c in enumerate(chars)}
    
    def encode_chars(text):
        text = str(text)
        encoded = [char_to_id.get(c, 1) for c in text[:max_char_len]]
        # Pad sequence to max_char_len
        return encoded + [0] * (max_char_len - len(encoded))

    # Pre-encode unique 'before' tokens to save computation
    print("Encoding character sequences...")
    all_before = pd.concat([X_train['before'], X_val['before'], X_test['before']]).unique()
    token_char_map = {t: encode_chars(t) for t in all_before}

    def process_split(df_in: pd.DataFrame, y_series: pd.Series = None) -> Tuple[X, y]:
        """
        Processes a split into a DataFrame of model-ready features.
        Maintains row alignment with original data.
        """
        n_samples = len(df_in)
        df = df_in.copy()
        df['orig_idx'] = np.arange(n_samples)
        
        # Sort to ensure sentence_id and token_id order for sequence modeling
        df_sorted = df.sort_values(['sentence_id', 'token_id'])
        
        # Group tokens by sentence_id
        sent_groups = df_sorted.groupby('sentence_id', sort=False)
        sentences = sent_groups['before'].apply(lambda x: [str(t) for t in x]).tolist()
        unique_sent_ids = df_sorted['sentence_id'].unique()
        sid_to_idx = {sid: i for i, sid in enumerate(unique_sent_ids)}
        
        # BERT Tokenization with word-level alignment
        encodings = tokenizer(
            sentences,
            is_split_into_words=True,
            padding='max_length',
            truncation=True,
            max_length=max_sent_len,
            return_tensors='np'
        )
        
        # Mappings from original tokens to BERT sub-word range
        word_starts = np.zeros(n_samples, dtype=np.int32)
        word_ends = np.zeros(n_samples, dtype=np.int32)
        
        sent_sizes = sent_groups.size().values
        offsets = np.concatenate(([0], np.cumsum(sent_sizes)[:-1]))
        
        for i in range(len(sentences)):
            wids = encodings.word_ids(batch_index=i)
            if wids is None: continue
            
            # Record start/end indices of sub-words for each original word
            bounds = {}
            for sub_idx, wid in enumerate(wids):
                if wid is not None:
                    if wid not in bounds: bounds[wid] = [sub_idx, sub_idx]
                    else: bounds[wid][1] = sub_idx
            
            base_off = offsets[i]
            s_size = sent_sizes[i]
            for wid, (start, end) in bounds.items():
                if wid < s_size:
                    word_starts[base_off + wid] = start
                    word_ends[base_off + wid] = end
        
        # Map sorted results back to original indices to maintain row alignment
        token_sent_idx = df_sorted['sentence_id'].map(sid_to_idx).values
        restore_idx = np.argsort(df_sorted['orig_idx'].values)
        
        # Construct the final feature DataFrame
        X_out = pd.DataFrame(index=df_in.index)
        
        # High-dimensional BERT features stored as object/list columns
        X_out['input_ids'] = list(encodings['input_ids'][token_sent_idx][restore_idx])
        X_out['attention_mask'] = list(encodings['attention_mask'][token_sent_idx][restore_idx])
        X_out['word_start'] = word_starts[restore_idx]
        X_out['word_end'] = word_ends[restore_idx]
        
        # Character-level and metadata features
        X_out['char_seq'] = [token_char_map[t] for t in df_in['before'].values]
        X_out['before'] = df_in['before'].values
        
        if 'class' in df_in.columns:
            X_out['class'] = df_in['class'].values
        else:
            X_out['class'] = ['PLAIN'] * n_samples
            
        y_out = y_series.values if y_series is not None else None
        return X_out, y_out

    # Execute transformation for all sets
    print("Processing Training set...")
    X_train_p, y_train_p = process_split(X_train, y_train)
    
    print("Processing Validation set...")
    X_val_p, y_val_p = process_split(X_val, y_val)
    
    print("Processing Test set...")
    X_test_p, _ = process_split(X_test)
    
    print(f"Preprocessing completed. Final counts: Train={len(X_train_p)}, Val={len(X_val_p)}, Test={len(X_test_p)}")
    return X_train_p, y_train_p, X_val_p, y_val_p, X_test_p