import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any
from transformers import AutoTokenizer, DebertaV2TokenizerFast
import re

# Task-adaptive type definitions
# Features are represented as a dictionary of NumPy arrays (input_ids, attention_mask)
# Targets are represented as a dictionary of NumPy arrays (start_positions, end_positions)
X = Dict[str, np.ndarray]
y = Dict[str, np.ndarray]

def find_best_span(text: str, selected_text: str) -> Tuple[int, int]:
    """
    Implements the Middle-Closest Heuristic to resolve multi-occurrence ambiguity.
    Finds all occurrences of selected_text in text and selects the one whose 
    midpoint is closest to the midpoint of the full text.
    """
    if not selected_text:
        return 0, 0
        
    # Find all start/end character indices where selected_text occurs in text.
    # Use re.escape to handle special characters in the selected_text.
    matches = [m.span() for m in re.finditer(re.escape(selected_text), text)]
    
    if not matches:
        # Fallback to simple find if regex fails (though EDA indicates 100% substring integrity)
        start = text.find(selected_text)
        if start == -1:
            return 0, 0
        return start, start + len(selected_text)
        
    # Midpoint of the full text
    text_mid = len(text) / 2
    
    best_span = matches[0]
    min_dist = float('inf')
    
    for start, end in matches:
        span_mid = (start + end) / 2
        dist = abs(span_mid - text_mid)
        if dist < min_dist:
            min_dist = dist
            best_span = (start, end)
            
    return best_span

def preprocess(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame
) -> Tuple[X, y, X, y, X]:
    """
    Transforms raw tweet data and sentiment labels into DeBERTa-ready features and span indices.
    
    This implementation uses DebertaV2TokenizerFast (v3-large) and maps character-level 
    'selected_text' substrings to token-level 'start_positions' and 'end_positions'.
    """
    
    # Technical Specification: DebertaV2TokenizerFast from 'microsoft/deberta-v3-large'
    # Use AutoTokenizer with use_fast=True to ensure the fast Rust-based implementation is loaded.
    print("Initializing DeBERTa-v3-large tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained('microsoft/deberta-v3-large', use_fast=True)
    
    # Technical Specification: max_length=160
    MAX_LEN = 160

    def _process_batch(df: pd.DataFrame, targets: pd.Series = None) -> Tuple[X, y]:
        """Helper to process a batch of data into encoded features and target spans."""
        num_samples = len(df)
        
        # Initialize feature buffers (int32 for efficiency and compatibility)
        input_ids = np.zeros((num_samples, MAX_LEN), dtype='int32')
        attention_mask = np.zeros((num_samples, MAX_LEN), dtype='int32')
        
        # Initialize target buffers if targets (selected_text) are provided
        start_positions = None
        end_positions = None
        if targets is not None:
            start_positions = np.zeros(num_samples, dtype='int32')
            end_positions = np.zeros(num_samples, dtype='int32')
        
        # Access underlying values to minimize overhead
        texts = df['text'].astype(str).values
        sentiments = df['sentiment'].astype(str).values
        selected_texts = targets.astype(str).values if targets is not None else None
        
        for i in range(num_samples):
            txt = texts[i]
            sent = sentiments[i]
            
            # Encode: [CLS] sentiment [SEP] text [SEP]
            # Tokenizer handles special token sequences and padding.
            encoded = tokenizer.encode_plus(
                sent,
                txt,
                add_special_tokens=True,
                max_length=MAX_LEN,
                padding='max_length',
                truncation=True,
                return_offsets_mapping=True
            )
            
            input_ids[i] = encoded['input_ids']
            attention_mask[i] = encoded['attention_mask']
            
            if targets is not None:
                sel = selected_texts[i]
                
                # Step 1: Locate the character span of selected_text using Middle-Closest heuristic.
                start_char, end_char = find_best_span(txt, sel)
                
                offsets = encoded['offset_mapping']
                # sequence_ids() identifies segments (0: sentiment, 1: text, None: special)
                sequence_ids = encoded.sequence_ids() 
                
                # Identify valid token indices for the 'text' segment (sequence_id == 1)
                text_token_indices = [idx for idx, s_id in enumerate(sequence_ids) if s_id == 1]
                
                if not text_token_indices:
                    # Fallback to 0 if text is entirely truncated
                    s_token, e_token = 0, 0
                else:
                    # Default to start and end of text segment
                    s_token = text_token_indices[0]
                    e_token = text_token_indices[-1]
                    
                    # Step 2: Map character indices to token indices within the 'text' segment
                    # DeBERTa-v2 uses SentencePiece; offsets align tokens with the original text.
                    for idx in text_token_indices:
                        off_start, off_end = offsets[idx]
                        
                        # Find the token that contains the start character
                        if off_start <= start_char < off_end:
                            s_token = idx
                        # Find the token that contains the end character
                        if off_start < end_char <= off_end:
                            e_token = idx
                    
                    # Ensure valid span boundaries
                    if e_token < s_token:
                        e_token = s_token
                
                start_positions[i] = s_token
                end_positions[i] = e_token
        
        # Structure outputs
        X_res = {
            'input_ids': input_ids, 
            'attention_mask': attention_mask
        }
        y_res = {
            'start_positions': start_positions, 
            'end_positions': end_positions
        } if targets is not None else None
        
        return X_res, y_res

    # Preprocessing pipeline: Fit-transform logic
    # Tokenizers are pre-trained, so 'fitting' corresponds to loading weights.
    print(f"Processing training set ({len(X_train)} samples)...")
    X_train_p, y_train_p = _process_batch(X_train, y_train)
    
    print(f"Processing validation set ({len(X_val)} samples)...")
    X_val_p, y_val_p = _process_batch(X_val, y_val)
    
    print(f"Processing test set ({len(X_test)} samples)...")
    X_test_p, _ = _process_batch(X_test)

    # Final validation of row alignment and output existence
    if any(v is None for v in (X_train_p, y_train_p, X_val_p, y_val_p, X_test_p)):
        raise ValueError("Preprocessing failure: Produced None values in expected output set.")

    print("Preprocessing complete. DeBERTa-ready features generated.")
    return X_train_p, y_train_p, X_val_p, y_val_p, X_test_p