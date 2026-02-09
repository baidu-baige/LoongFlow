import pandas as pd
import numpy as np
from transformers import AutoTokenizer
from typing import Tuple, List, Dict, Any

# Target paths
BASE_DATA_PATH = "/mnt/pfs/loongflow/devmachine/h20-12/evolux/output/mlebench/chaii-hindi-and-tamil-question-answering/prepared/public"
OUTPUT_DATA_PATH = "output/af0f7d71-a062-46e3-8926-51aedd28d3b4/3/executor/output"

# Task-adaptive type definitions
X = pd.DataFrame
y = pd.DataFrame

def _get_token_position(
    offsets: List[Tuple[int, int]], 
    sequence_ids: List[int], 
    start_char: int, 
    end_char: int, 
    cls_index: int
) -> Tuple[int, int]:
    """
    Maps character start/end positions to token start/end positions within a chunk.
    If the answer is not fully contained in the context part of the chunk, returns (cls_index, cls_index).
    """
    # Find the start and end of the context within the tokens
    try:
        token_start_index = 0
        while sequence_ids[token_start_index] != 1:
            token_start_index += 1

        token_end_index = len(sequence_ids) - 1
        while sequence_ids[token_end_index] != 1:
            token_end_index -= 1
    except (IndexError, ValueError):
        return cls_index, cls_index

    # Check if the answer is fully contained within this chunk's context
    if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
        return cls_index, cls_index
    
    # Map character positions to token positions
    # Start token: first token that starts at or after start_char
    curr_start = token_start_index
    while curr_start <= token_end_index and offsets[curr_start][0] < start_char:
        curr_start += 1
    
    # End token: last token that ends at or before end_char
    curr_end = token_end_index
    while curr_end >= token_start_index and offsets[curr_end][1] > end_char:
        curr_end -= 1
        
    return curr_start, curr_end

def _prepare_dual_path_data(
    X_df: pd.DataFrame, 
    y_df: pd.DataFrame, 
    tok_xlmr: Any, 
    tok_muril: Any, 
    max_length: int, 
    doc_stride: int,
    is_train: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generates model-ready chunks using a dual-path tokenization strategy.
    Uses XLM-RoBERTa as the primary chunker to define context windows,
    then aligns MuRIL tokens to those same windows.
    """
    # 1. Primary tokenization with XLM-R (Sliding Window)
    xlmr_encodings = tok_xlmr(
        X_df["question"].tolist(),
        X_df["context"].tolist(),
        truncation="only_second",
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_mapping = xlmr_encodings.pop("overflow_to_sample_mapping")
    xlmr_offsets = xlmr_encodings["offset_mapping"]
    
    # 2. Extract context spans for MuRIL alignment
    # We want MuRIL to tokenize the same question + context-substring as XLM-R
    chunk_questions = []
    chunk_sub_contexts = []
    chunk_original_offsets = [] # Stores (start_char, end_char) of the context window relative to original context
    
    for i in range(len(xlmr_encodings["input_ids"])):
        sample_idx = sample_mapping[i]
        seq_ids = xlmr_encodings.sequence_ids(i)
        offsets = xlmr_offsets[i]
        
        # Identify the character range of the context in this chunk
        ctx_tokens = [j for j, s_id in enumerate(seq_ids) if s_id == 1]
        if not ctx_tokens:
            # Fallback if no context tokens (shouldn't happen with truncation='only_second')
            chunk_sub_contexts.append("")
            chunk_original_offsets.append((0, 0))
        else:
            c_start = offsets[ctx_tokens[0]][0]
            c_end = offsets[ctx_tokens[-1]][1]
            chunk_sub_contexts.append(X_df.iloc[sample_idx]["context"][c_start:c_end])
            chunk_original_offsets.append((c_start, c_end))
        
        chunk_questions.append(X_df.iloc[sample_idx]["question"])

    # 3. Secondary tokenization with MuRIL (Aligned to XLM-R spans)
    muril_encodings = tok_muril(
        chunk_questions,
        chunk_sub_contexts,
        truncation="only_second", # In case the sub_context is too dense for MuRIL
        max_length=max_length,
        padding="max_length",
        return_offsets_mapping=True,
    )
    muril_offsets = muril_encodings["offset_mapping"]

    # 4. Feature and Target Construction
    features = []
    targets = []
    
    xlmr_cls_id = tok_xlmr.cls_token_id
    muril_cls_id = tok_muril.cls_token_id

    for i in range(len(xlmr_encodings["input_ids"])):
        sample_idx = sample_mapping[i]
        
        # Feature collection
        feat = {
            "example_id": X_df.iloc[sample_idx]["id"],
            "language": X_df.iloc[sample_idx]["language"],
            "xlmr_input_ids": np.array(xlmr_encodings["input_ids"][i], dtype=np.int32),
            "xlmr_attention_mask": np.array(xlmr_encodings["attention_mask"][i], dtype=np.int8),
            "xlmr_offset_mapping": xlmr_offsets[i],
            "muril_input_ids": np.array(muril_encodings["input_ids"][i], dtype=np.int32),
            "muril_attention_mask": np.array(muril_encodings["attention_mask"][i], dtype=np.int8),
            "muril_offset_mapping": muril_offsets[i],
        }
        features.append(feat)

        if is_train and y_df is not None:
            ans_start_orig = y_df.iloc[sample_idx]["answer_start"]
            ans_text = str(y_df.iloc[sample_idx]["answer_text"])
            ans_end_orig = ans_start_orig + len(ans_text)
            
            # XLM-R positions (relative to original context)
            xlmr_start, xlmr_end = _get_token_position(
                xlmr_offsets[i], 
                xlmr_encodings.sequence_ids(i), 
                ans_start_orig, 
                ans_end_orig, 
                xlmr_encodings["input_ids"][i].index(xlmr_cls_id)
            )
            
            # MuRIL positions (relative to sub-context)
            chunk_c_start, chunk_c_end = chunk_original_offsets[i]
            # Map original char indices to chunk-local char indices
            ans_start_local = ans_start_orig - chunk_c_start
            ans_end_local = ans_end_orig - chunk_c_start
            
            muril_start, muril_end = _get_token_position(
                muril_offsets[i],
                muril_encodings.sequence_ids(i),
                ans_start_local,
                ans_end_local,
                muril_encodings["input_ids"][i].index(muril_cls_id)
            )
            
            targets.append({
                "xlmr_start_positions": xlmr_start,
                "xlmr_end_positions": xlmr_end,
                "muril_start_positions": muril_start,
                "muril_end_positions": muril_end
            })

    X_processed = pd.DataFrame(features)
    y_processed = pd.DataFrame(targets) if targets else pd.DataFrame()
    
    return X_processed, y_processed

def preprocess(
    X_train: X,
    y_train: y,
    X_val: X,
    y_val: y,
    X_test: X
) -> Tuple[X, y, X, y, X]:
    """
    Transforms raw data into dual-path model-ready format for MuRIL and XLM-Roberta.
    """
    print("Initializing dual-path preprocessing (MuRIL-Large + XLM-R-Large)...")
    
    # Configuration
    xlmr_model = "xlm-roberta-large"
    muril_model = "google/muril-large-cased"
    max_length = 512
    doc_stride = 128
    
    print(f"Loading tokenizers: {xlmr_model} and {muril_model}")
    tok_xlmr = AutoTokenizer.from_pretrained(xlmr_model, use_fast=True)
    tok_muril = AutoTokenizer.from_pretrained(muril_model, use_fast=True)
    
    # Process Train
    print(f"Processing training set (Original: {len(X_train)} samples)...")
    X_train_proc, y_train_proc = _prepare_dual_path_data(
        X_train, y_train, tok_xlmr, tok_muril, max_length, doc_stride, is_train=True
    )
    
    # Process Val
    print(f"Processing validation set (Original: {len(X_val)} samples)...")
    X_val_proc, y_val_proc = _prepare_dual_path_data(
        X_val, y_val, tok_xlmr, tok_muril, max_length, doc_stride, is_train=True
    )
    
    # Process Test
    print(f"Processing test set (Original: {len(X_test)} samples)...")
    X_test_proc, _ = _prepare_dual_path_data(
        X_test, None, tok_xlmr, tok_muril, max_length, doc_stride, is_train=False
    )
    
    # Ensure Test completeness
    missing_ids = set(X_test["id"]) - set(X_test_proc["example_id"])
    if missing_ids:
        raise ValueError(f"Preprocessing dropped test samples: {missing_ids}")
        
    print("Preprocessing complete.")
    print(f"Final chunk counts - Train: {len(X_train_proc)}, Val: {len(X_val_proc)}, Test: {len(X_test_proc)}")
    
    return (
        X_train_proc, 
        y_train_proc, 
        X_val_proc, 
        y_val_proc, 
        X_test_proc
    )