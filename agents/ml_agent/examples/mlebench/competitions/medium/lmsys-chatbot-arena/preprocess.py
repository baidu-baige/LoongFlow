import pandas as pd
import numpy as np
import os
from typing import Tuple, Dict, Any
from transformers import AutoTokenizer
from datasets import Dataset

# Task-adaptive type definitions
# X: pd.DataFrame containing 'input_ids' and 'attention_mask' columns (each cell is a list of tokens)
# y: np.ndarray of integer targets [0, 1, 2]
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
    Transforms raw text data into model-ready numerical tensors using DeBERTa-v3-large tokenization.
    
    Implements a specific truncation strategy to maximize information density:
    - Prompt: Head-only truncation (256 tokens).
    - Responses: Head+Tail truncation (640 tokens total: 192 head, 448 tail).
    - Format: [CLS] prompt [SEP] response_a [SEP] response_b [SEP]
    """
    # 1. Technical Configuration
    MODEL_NAME = "microsoft/deberta-v3-large"
    MAX_LENGTH = 1536
    PROMPT_MAX = 256
    RESP_MAX = 640
    RESP_HEAD = 192  # 30% of 640
    RESP_TAIL = 448  # 70% of 640
    NUM_PROC = 36    # Leverage all 36 cores
    
    print(f"Initializing Preprocessing with {MODEL_NAME}...")
    
    # Disable parallelism in the tokenizer itself to avoid deadlocks when using multi-processing map
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    
    # Special token IDs for DeBERTa-v3
    CLS_ID = tokenizer.cls_token_id
    SEP_ID = tokenizer.sep_token_id
    PAD_ID = tokenizer.pad_token_id

    # 2. Define Batch Tokenization Logic
    def tokenize_batch(examples):
        """
        Processes a batch of text samples into tokenized tensors.
        """
        batch_input_ids = []
        batch_attention_mask = []
        
        # Ensure all inputs are strings to prevent failures on missing data
        prompts = [str(x) if x is not None else "" for x in examples['prompt']]
        resps_a = [str(x) if x is not None else "" for x in examples['response_a']]
        resps_b = [str(x) if x is not None else "" for x in examples['response_b']]
        
        for p, ra, rb in zip(prompts, resps_a, resps_b):
            # Step A: Tokenize components individually (no special tokens yet)
            p_ids = tokenizer.encode(p, add_special_tokens=False)
            ra_ids = tokenizer.encode(ra, add_special_tokens=False)
            rb_ids = tokenizer.encode(rb, add_special_tokens=False)
            
            # Step B: Apply intelligent truncation
            # Prompt: Head-only
            p_ids = p_ids[:PROMPT_MAX]
            
            # Responses: Head (30%) + Tail (70%)
            def apply_ht_trunc(ids, head, tail):
                if len(ids) <= (head + tail):
                    return ids
                return ids[:head] + ids[-tail:]
            
            ra_ids = apply_ht_trunc(ra_ids, RESP_HEAD, RESP_TAIL)
            rb_ids = apply_ht_trunc(rb_ids, RESP_HEAD, RESP_TAIL)
            
            # Step C: Construct sequence: [CLS] prompt [SEP] response_a [SEP] response_b [SEP]
            combined = [CLS_ID] + p_ids + [SEP_ID] + ra_ids + [SEP_ID] + rb_ids + [SEP_ID]
            
            # Step D: Final constraint to global context window (MAX_LENGTH)
            combined = combined[:MAX_LENGTH]
            
            # Step E: Create Attention Mask and apply Padding
            mask = [1] * len(combined)
            pad_len = MAX_LENGTH - len(combined)
            if pad_len > 0:
                combined += [PAD_ID] * pad_len
                mask += [0] * pad_len
                
            batch_input_ids.append(combined)
            batch_attention_mask.append(mask)
            
        return {
            "input_ids": batch_input_ids,
            "attention_mask": batch_attention_mask
        }

    # 3. Execution Wrapper using HuggingFace Datasets for parallelism
    def process_dataframe(df: pd.DataFrame, name: str) -> X:
        print(f"  Tokenizing {name} set (Size: {len(df)})...")
        # Optimization: Only load columns needed for tokenization
        ds = Dataset.from_pandas(df[['prompt', 'response_a', 'response_b']].reset_index(drop=True))
        
        processed_ds = ds.map(
            tokenize_batch,
            batched=True,
            batch_size=1000,
            num_proc=NUM_PROC,
            remove_columns=ds.column_names,
            desc=f"Parallel Tokenization: {name}"
        )
        
        # Return as a DataFrame to ensure len(X) == number of samples
        # Columns: 'input_ids' (List[int]), 'attention_mask' (List[int])
        return pd.DataFrame({
            "input_ids": processed_ds["input_ids"],
            "attention_mask": processed_ds["attention_mask"]
        })

    # 4. Process all splits
    X_train_processed = process_dataframe(X_train, "train")
    y_train_processed = y_train.values

    X_val_processed = process_dataframe(X_val, "validation")
    y_val_processed = y_val.values

    X_test_processed = process_dataframe(X_test, "test")

    # 5. Validation and Consistency Checks
    # Verify sample counts and alignment
    if len(X_train_processed) != len(y_train_processed):
        raise ValueError(f"Alignment Error: X_train ({len(X_train_processed)}) != y_train ({len(y_train_processed)})")
    if len(X_val_processed) != len(y_val_processed):
        raise ValueError(f"Alignment Error: X_val ({len(X_val_processed)}) != y_val ({len(y_val_processed)})")
    if len(X_test_processed) != len(X_test):
        raise ValueError(f"Alignment Error: X_test ({len(X_test_processed)}) != X_test source ({len(X_test)})")
        
    print("Preprocessing successfully completed.")
    return (
        X_train_processed,
        y_train_processed,
        X_val_processed,
        y_val_processed,
        X_test_processed
    )