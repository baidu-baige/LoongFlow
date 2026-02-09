import numpy as np
import pandas as pd
import os
from typing import Dict, List, Any
from transformers import AutoTokenizer

# Task-adaptive type definitions
# y: Ground truth targets as dictionary of span indices (from preprocess stage)
y = Dict[str, np.ndarray]
# Predictions (Input): Dictionary of logits (from train_and_predict stage)
# Predictions (Output): Array of final predicted selected_text strings
Predictions = Any

BASE_DATA_PATH = "/mnt/pfs/loongflow/devmachine/1-04/evolux/output/mlebench/tweet-sentiment-extraction/prepared/public"
OUTPUT_DATA_PATH = "output/1cc1b6ac-193c-4f3f-a388-380b832f53e8/5/executor/output"

def ensemble(
    all_val_preds: Dict[str, Predictions],
    all_test_preds: Dict[str, Predictions],
    y_val: y,
) -> np.ndarray:
    """
    Combines predictions from multiple models (folds) using Logit Averaging and applies
    high-confidence heuristics for Neutral and Short Text samples.

    Strategy:
    1. Average 'start_logits' and 'end_logits' across all provided models/folds.
    2. Apply Heuristics:
       - If sentiment is 'neutral', return the full text.
       - If text has 3 or fewer words, return the full text.
    3. Decode: For other samples, find the token span (s, e) that maximizes 
       (start_logits[s] + end_logits[e]) within the text segment, where s <= e.
    """
    if not all_test_preds:
        raise ValueError("Ensemble received empty all_test_preds. Ensure models were trained and predicted successfully.")

    model_keys = list(all_test_preds.keys())
    print(f"Ensembling {len(model_keys)} models...")

    # Step 1: Logit Averaging
    # Extract dimensions from the first model's results
    num_samples = all_test_preds[model_keys[0]]['start_logits'].shape[0]
    seq_len = all_test_preds[model_keys[0]]['start_logits'].shape[1]
    
    avg_start_logits = np.zeros((num_samples, seq_len), dtype=np.float32)
    avg_end_logits = np.zeros((num_samples, seq_len), dtype=np.float32)
    
    for key in model_keys:
        avg_start_logits += all_test_preds[key]['start_logits']
        avg_end_logits += all_test_preds[key]['end_logits']
    
    avg_start_logits /= len(model_keys)
    avg_end_logits /= len(model_keys)

    # Step 2: Load Test Metadata for Heuristics and Decoding
    # Follows load_data.py specification to ensure character-level alignment integrity
    test_path = os.path.join(BASE_DATA_PATH, "test.csv")
    try:
        test_df = pd.read_csv(test_path, keep_default_na=False, dtype=str)
    except Exception as e:
        print(f"Failed to load test meta-data: {e}")
        raise

    # Subset test metadata if validation_mode was active in upstream stages
    if len(test_df) > num_samples:
        print(f"Subsetting test metadata from {len(test_df)} to {num_samples} samples.")
        test_df = test_df.head(num_samples)
    elif len(test_df) < num_samples:
        raise ValueError(f"Logit count ({num_samples}) exceeds available test metadata rows ({len(test_df)}).")

    # Step 3: Initialize Tokenizer
    # Must match the model architecture and preprocessing used in Stage 3: microsoft/deberta-v3-large
    print("Loading DeBERTa-v3-large tokenizer for span decoding...")
    tokenizer = AutoTokenizer.from_pretrained('microsoft/deberta-v3-large', use_fast=True)
    MAX_LEN = 160

    final_predictions = []

    # Step 4: Final Prediction Generation with Dual-Heuristic Override
    print("Applying Ensemble Heuristics and Decoding Spans...")
    for i in range(num_samples):
        text = test_df.iloc[i]['text']
        sentiment = test_df.iloc[i]['sentiment']
        
        # TECHNICAL SPECIFICATION: Dual-Heuristic Override
        # 1. Neutral Tweets: Identity mapping is near-optimal (Jaccard ~0.98)
        # 2. Short Tweets: Words <= 3 often favor identity mapping (Jaccard 0.87-0.94)
        if sentiment.strip().lower() == 'neutral' or len(text.split()) <= 3:
            final_predictions.append(text)
            continue

        # TECHNICAL SPECIFICATION: Logit Decoding for other samples
        # Find the span (s, e) that maximizes start_logits[s] + end_logits[e] where s <= e
        
        # Re-tokenize to identify text segment indices precisely
        encoded = tokenizer.encode_plus(
            sentiment,
            text,
            add_special_tokens=True,
            max_length=MAX_LEN,
            padding='max_length',
            truncation=True,
            return_offsets_mapping=True
        )
        
        offsets = encoded['offset_mapping']
        sequence_ids = encoded.sequence_ids()
        
        # Identify token indices belonging to the 'text' segment (sequence_id == 1)
        # Sequence format: [CLS] sentiment [SEP] text [SEP]
        text_token_indices = [idx for idx, s_id in enumerate(sequence_ids) if s_id == 1]
        
        if not text_token_indices:
            # Fallback for edge cases where text might be truncated or empty
            final_predictions.append(text)
            continue
            
        best_score = -float('inf')
        best_start = text_token_indices[0]
        best_end = text_token_indices[0]
        
        # Search for optimal span restricted to text tokens
        for s_idx in text_token_indices:
            s_logit = avg_start_logits[i, s_idx]
            for e_idx in text_token_indices:
                if e_idx >= s_idx:
                    combined_score = s_logit + avg_end_logits[i, e_idx]
                    if combined_score > best_score:
                        best_score = combined_score
                        best_start = s_idx
                        best_end = e_idx
        
        # Map token span back to original character indices
        char_start = offsets[best_start][0]
        char_end = offsets[best_end][1]
        
        # Extract the exact supported phrase including punctuation and whitespace
        selected_text = text[char_start:char_end]
        final_predictions.append(selected_text)

    # Verification of output alignment
    ensemble_output = np.array(final_predictions)
    if len(ensemble_output) != num_samples:
        raise RuntimeError(f"Ensemble size mismatch: Expected {num_samples}, got {len(ensemble_output)}")
    
    # Ensure no NaN or None values leaked into the final predictions
    if any(p is None for p in ensemble_output):
        raise ValueError("Ensemble generated None values in the output prediction array.")

    print(f"Ensemble complete. Generated {len(ensemble_output)} final predictions.")
    return ensemble_output