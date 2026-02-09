import os
import numpy as np
import pandas as pd
from transformers import RobertaTokenizer, BertTokenizer
from joblib import Parallel, delayed
from typing import Tuple

# Task-adaptive type definitions
X = pd.DataFrame      # Feature matrix containing 2067 columns (IDs, Masks, Metadata, Structural Features)
y = pd.DataFrame      # Target vector containing 30 continuous labels

BASE_DATA_PATH = "/mnt/pfs/loongflow/devmachine/2-05/evolux/output/mlebench/google-quest-challenge/prepared/public"
OUTPUT_DATA_PATH = "output/aaa741b3-cb02-44fc-a666-dd434e563444/8/executor/output"

def get_structural_features(title: str, body: str, answer: str) -> list:
    """Extracts 14 split structural features for Question and Answer."""
    q_text = str(title) + " " + str(body)
    a_text = str(answer)
    
    tags = ["<code>", "<a>", "<ul>", "<li>", "?", "!"]
    
    # Question structural features
    q_feats = [q_text.count(t) for t in tags]
    q_caps = sum(1 for c in q_text if c.isupper()) / (len(q_text) + 1e-7)
    q_feats.append(q_caps)
    
    # Answer structural features
    a_feats = [a_text.count(t) for t in tags]
    a_caps = sum(1 for c in a_text if c.isupper()) / (len(a_text) + 1e-7)
    a_feats.append(a_caps)
    
    return q_feats + a_feats

def get_tokens(tokenizer, title: str, body: str, answer: str, model_type: str) -> Tuple[list, list]:
    """Helper to tokenize a row for a specific model type (RoBERTa or BERT)."""
    # Use spaces for RoBERTa as it is BPE based and sensitive to starting tokens
    if model_type == 'roberta':
        t_toks = tokenizer.tokenize(" " + str(title))[:64]
        b_all = tokenizer.tokenize(" " + str(body))
        a_all = tokenizer.tokenize(" " + str(answer))
    else:
        t_toks = tokenizer.tokenize(str(title))[:64]
        b_all = tokenizer.tokenize(str(body))
        a_all = tokenizer.tokenize(str(answer))
        
    def trunc_bt(toks):
        # 172 Head + 50 Tail logic
        if len(toks) > 222:
            return toks[:172] + toks[-50:]
        return toks
        
    b_toks = trunc_bt(b_all)
    a_toks = trunc_bt(a_all)
    
    # Structure: [CLS] Title [SEP] Body [SEP] Answer [SEP]
    # Using internal attributes to ensure correct mapping for both BERT and RoBERTa
    tokens = [tokenizer.cls_token] + t_toks + [tokenizer.sep_token] + b_toks + \
             [tokenizer.sep_token] + a_toks + [tokenizer.sep_token]
    
    ids = tokenizer.convert_tokens_to_ids(tokens)
    ids = ids[:512]
    mask = [1] * len(ids)
    
    # Padding
    padding_len = 512 - len(ids)
    ids = ids + [tokenizer.pad_token_id] * padding_len
    mask = mask + [0] * padding_len
    
    return ids, mask

def tokenize_row(title, body, answer, category, host, cat_map, host_map, r_tokenizer, b_tokenizer) -> list:
    """Processes a single row into the full feature vector of 2067 elements."""
    # 1. RoBERTa IDs/Masks (1024)
    r_ids, r_masks = get_tokens(r_tokenizer, title, body, answer, 'roberta')
    
    # 2. BERT IDs/Masks (1024)
    b_ids, b_masks = get_tokens(b_tokenizer, title, body, answer, 'bert')
    
    # 3. Metadata (category_id, host_id)
    c_idx = cat_map.get(category, len(cat_map))
    h_idx = host_map.get(host, len(host_map))
    
    # 4. Lengths (3)
    t_len = len(str(title)) / 150
    b_len = len(str(body)) / 20000
    a_len = len(str(answer)) / 23000
    
    # 5. Structural Features (14)
    struct = get_structural_features(title, body, answer)
    
    return r_ids + r_masks + b_ids + b_masks + [c_idx, h_idx, t_len, b_len, a_len] + struct

def preprocess(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_val: pd.DataFrame,
    X_test: pd.DataFrame
) -> Tuple[X, y, X, y, X]:
    """
    Transforms raw data into model-ready format for a dual-model ensemble.
    Includes RoBERTa/BERT tokenization, metadata encoding, and split structural features.
    """
    print("Execution Stage: preprocess")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Step 1: Initialize Tokenizers with fallback logic
    r_local_path = '/mnt/pfs/loongflow/devmachine/2-05/models/roberta-base'
    b_local_path = '/mnt/pfs/loongflow/devmachine/2-05/models/bert-base-uncased'
    
    r_model = r_local_path if os.path.exists(r_local_path) else 'roberta-base'
    b_model = b_local_path if os.path.exists(b_local_path) else 'bert-base-uncased'
    
    print(f"Loading RoBERTa tokenizer from: {r_model}")
    r_tokenizer = RobertaTokenizer.from_pretrained(r_model)
    print(f"Loading BERT tokenizer from: {b_model}")
    b_tokenizer = BertTokenizer.from_pretrained(b_model)

    # Step 2: Fit Transformers on X_train ONLY
    categories = sorted(X_train['category'].unique())
    cat_map = {val: i for i, val in enumerate(categories)}
    
    hosts = sorted(X_train['host'].unique())
    host_map = {val: i for i, val in enumerate(hosts)}

    def process_split(df: pd.DataFrame) -> pd.DataFrame:
        """Parallelizes the tokenization and feature extraction across the dataframe."""
        results = Parallel(n_jobs=36)(
            delayed(tokenize_row)(
                row.question_title, 
                row.question_body, 
                row.answer, 
                row.category, 
                row.host,
                cat_map,
                host_map,
                r_tokenizer,
                b_tokenizer
            ) for row in df.itertuples(index=False)
        )
        
        col_names = [f'r_id_{i}' for i in range(512)] + \
                    [f'r_mask_{i}' for i in range(512)] + \
                    [f'b_id_{i}' for i in range(512)] + \
                    [f'b_mask_{i}' for i in range(512)] + \
                    ['category_id', 'host_id', 'title_len', 'body_len', 'answer_len'] + \
                    ['q_code', 'q_a', 'q_ul', 'q_li', 'q_ques', 'q_excl', 'q_cap'] + \
                    ['a_code', 'a_a', 'a_ul', 'a_li', 'a_ques', 'a_excl', 'a_cap']
        
        processed_df = pd.DataFrame(results, index=df.index, columns=col_names)
        return processed_df.astype(np.float32)

    # Step 3: Transform Train, Val, and Test sets
    print(f"Preprocessing Training set (rows: {len(X_train)})...")
    X_train_processed = process_split(X_train)
    y_train_processed = y_train.astype(np.float32)

    print(f"Preprocessing Validation set (rows: {len(X_val)})...")
    X_val_processed = process_split(X_val)
    y_val_processed = y_val.astype(np.float32)

    print(f"Preprocessing Test set (rows: {len(X_test)})...")
    X_test_processed = process_split(X_test)

    # Step 4: Final Validation and Cleanup
    datasets = [
        ("Train", X_train_processed), 
        ("Val", X_val_processed), 
        ("Test", X_test_processed)
    ]
    
    for name, df in datasets:
        if df.isna().any().any():
            print(f"Warning: {name} features contain NaN. Filling with 0.")
            df.fillna(0, inplace=True)
        if np.isinf(df.values).any():
            print(f"Warning: {name} features contain Inf. Replacing with 0.")
            df.replace([np.inf, -np.inf], 0, inplace=True)
            
    if len(X_train_processed) != len(y_train_processed):
        raise ValueError(f"Train alignment error: X({len(X_train_processed)}) != y({len(y_train_processed)})")
    if len(X_val_processed) != len(y_val_processed):
        raise ValueError(f"Val alignment error: X({len(X_val_processed)}) != y({len(y_val_processed)})")
    if len(X_test_processed) != len(X_test):
        raise ValueError(f"Test completeness error: X({len(X_test_processed)}) != Original({len(X_test)})")

    print(f"Preprocessing complete. Total features per sample: {X_train_processed.shape[1]}")
    return X_train_processed, y_train_processed, X_val_processed, y_val_processed, X_test_processed