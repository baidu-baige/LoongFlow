import os
import string
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple
from collections import defaultdict
from transformers import AutoTokenizer

# Target paths
BASE_DATA_PATH = "/mnt/pfs/loongflow/devmachine/h20-12/evolux/output/mlebench/chaii-hindi-and-tamil-question-answering/prepared/public"
OUTPUT_DATA_PATH = "output/af0f7d71-a062-46e3-8926-51aedd28d3b4/3/executor/output"

# Type definitions
y = pd.DataFrame
Predictions = pd.DataFrame

def clean_prediction(text: str) -> str:
    """
    Applies aggressive whitespace stripping and punctuation cleaning to the answer string.
    This maximizes the word-level Jaccard score by removing boundary noise.
    """
    if not isinstance(text, str) or not text:
        return ""
    
    # Standard punctuation plus Hindi sentence markers (Purna Viram)। and (Deep Purna Viram)॥
    bad_chars = string.punctuation + "।" + "॥" + "—" + "–"
    return text.strip(bad_chars).strip()

def jaccard(str1: str, str2: str) -> float:
    """Computes the word-level Jaccard score between two strings."""
    a = set(str(str1).lower().split())
    b = set(str(str2).lower().split())
    c = a.intersection(b)
    if not a and not b: return 1.0
    return float(len(c)) / (len(a) + len(b) - len(c))

def get_mapping_data(X_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Replicates the preprocessing logic to extract offset mappings and sequence IDs.
    This ensures logits are correctly mapped back to character spans in the original context.
    """
    xlmr_model = "xlm-roberta-large"
    muril_model = "google/muril-large-cased"
    max_length = 512
    doc_stride = 128
    
    print(f"Loading tokenizers for mapping: {xlmr_model}, {muril_model}")
    tok_xlmr = AutoTokenizer.from_pretrained(xlmr_model, use_fast=True)
    tok_muril = AutoTokenizer.from_pretrained(muril_model, use_fast=True)
    
    # 1. XLM-R tokenization (Primary Chunking)
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
    chunk_questions = []
    chunk_sub_contexts = []
    chunk_original_offsets = []
    
    for i in range(len(xlmr_encodings["input_ids"])):
        sample_idx = sample_mapping[i]
        seq_ids = xlmr_encodings.sequence_ids(i)
        offsets = xlmr_offsets[i]
        
        ctx_tokens = [j for j, s_id in enumerate(seq_ids) if s_id == 1]
        if not ctx_tokens:
            chunk_sub_contexts.append("")
            chunk_original_offsets.append((0, 0))
        else:
            c_start = offsets[ctx_tokens[0]][0]
            c_end = offsets[ctx_tokens[-1]][1]
            chunk_sub_contexts.append(X_df.iloc[sample_idx]["context"][c_start:c_end])
            chunk_original_offsets.append((c_start, c_end))
        chunk_questions.append(X_df.iloc[sample_idx]["question"])

    # 3. MuRIL tokenization (Aligned to XLM-R chunks)
    muril_encodings = tok_muril(
        chunk_questions,
        chunk_sub_contexts,
        truncation="only_second",
        max_length=max_length,
        padding="max_length",
        return_offsets_mapping=True,
    )
    
    return {
        "sample_mapping": sample_mapping,
        "xlmr_offsets": xlmr_offsets,
        "xlmr_seq_ids": [xlmr_encodings.sequence_ids(i) for i in range(len(xlmr_encodings["input_ids"]))],
        "muril_offsets": muril_encodings["offset_mapping"],
        "muril_seq_ids": [muril_encodings.sequence_ids(i) for i in range(len(muril_encodings["input_ids"]))],
        "chunk_original_offsets": chunk_original_offsets
    }

def aggregate_scores(
    logits: np.ndarray,
    mappings: Dict[str, Any],
    model_type: str,
    n_best_size: int,
    max_answer_length: int
) -> Dict[int, Dict[Tuple[int, int], float]]:
    """
    Calculates span scores for a specific model architecture and maps them to character offsets.
    """
    sample_mapping = mappings["sample_mapping"]
    if model_type == "muril":
        offsets = mappings["muril_offsets"]
        seq_ids_list = mappings["muril_seq_ids"]
        chunk_orig_starts = [o[0] for o in mappings["chunk_original_offsets"]]
    else:
        offsets = mappings["xlmr_offsets"]
        seq_ids_list = mappings["xlmr_seq_ids"]
        chunk_orig_starts = [0] * len(sample_mapping) # XLM-R offsets are already global

    # results[sample_idx][(char_start, char_end)] = max_score_for_this_span
    model_results = defaultdict(lambda: defaultdict(lambda: -1e9))

    for i in range(len(logits)):
        sample_idx = sample_mapping[i]
        start_logits = logits[i, :, 0]
        end_logits = logits[i, :, 1]
        chunk_offsets = offsets[i]
        seq_ids = seq_ids_list[i]
        orig_offset = chunk_orig_starts[i]
        
        # Filter context indices
        context_indices = [idx for idx, s_id in enumerate(seq_ids) if s_id == 1]
        if not context_indices: continue

        # Get top candidates
        start_indexes = np.argsort(start_logits)[::-1][:n_best_size * 2]
        end_indexes = np.argsort(end_logits)[::-1][:n_best_size * 2]

        for start_index in start_indexes:
            if start_index not in context_indices: continue
            for end_index in end_indexes:
                if end_index not in context_indices: continue
                if end_index < start_index or end_index - start_index + 1 > max_answer_length:
                    continue
                
                score = start_logits[start_index] + end_logits[end_index]
                char_start = chunk_offsets[start_index][0] + orig_offset
                char_end = chunk_offsets[end_index][1] + orig_offset
                
                if score > model_results[sample_idx][(char_start, char_end)]:
                    model_results[sample_idx][(char_start, char_end)] = score
                    
    return model_results

def ensemble(
    all_val_preds: Dict[str, np.ndarray],
    all_test_preds: Dict[str, np.ndarray],
    y_val: pd.DataFrame,
) -> pd.DataFrame:
    """
    Combines predictions from MuRIL and XLM-RoBERTa using weighted span-score aggregation.
    """
    print(f"Starting ensemble process with {len(all_test_preds)} models...")
    
    # 1. Load Data
    from load_data import load_data
    X_train, y_train_full, X_test, test_ids = load_data()
    
    # 2. Extract mappings for test set
    test_mappings = get_mapping_data(X_test)
    
    # 3. Aggregate Architecture-specific Logits
    muril_test_list = [v for k, v in all_test_preds.items() if 'muril' in k.lower()]
    xlmr_test_list = [v for k, v in all_test_preds.items() if 'xlmr' in k.lower()]
    
    avg_muril_test = np.mean(muril_test_list, axis=0) if muril_test_list else None
    avg_xlmr_test = np.mean(xlmr_test_list, axis=0) if xlmr_test_list else None
    
    # 4. Parameters
    muril_weight = 0.6
    xlmr_weight = 0.4
    n_best_size = 20
    max_answer_length = 30
    
    # 5. Process Span Scores
    all_span_scores = defaultdict(lambda: defaultdict(float))
    
    if avg_muril_test is not None:
        print("Processing MuRIL test spans...")
        muril_spans = aggregate_scores(avg_muril_test, test_mappings, "muril", n_best_size, max_answer_length)
        w = muril_weight if avg_xlmr_test is not None else 1.0
        for s_idx, spans in muril_spans.items():
            for span, score in spans.items():
                all_span_scores[s_idx][span] += w * score
                
    if avg_xlmr_test is not None:
        print("Processing XLM-RoBERTa test spans...")
        xlmr_spans = aggregate_scores(avg_xlmr_test, test_mappings, "xlmr", n_best_size, max_answer_length)
        w = xlmr_weight if avg_muril_test is not None else 1.0
        for s_idx, spans in xlmr_spans.items():
            for span, score in spans.items():
                all_span_scores[s_idx][span] += w * score

    # 6. Generate Final Predictions
    print("Selecting best spans and applying post-processing...")
    final_predictions = []
    for i in range(len(X_test)):
        context = X_test.iloc[i]["context"]
        sample_spans = all_span_scores.get(i, {})
        
        if not sample_spans:
            final_predictions.append("")
            continue
            
        best_span = max(sample_spans.items(), key=lambda x: x[1])[0]
        raw_answer = context[best_span[0]:best_span[1]]
        cleaned_answer = clean_prediction(raw_answer)
        final_predictions.append(cleaned_answer)

    # 7. Construct Submission
    submission_df = pd.DataFrame({
        "id": test_ids.values,
        "PredictionString": final_predictions
    })
    
    # Validation Evaluation (Optional / Debugging)
    if all_val_preds and not y_val.empty:
        print("Note: Validation logit aggregation received. Calculating OOF performance is skipped to focus on test submission efficiency.")
    
    # Ensure integrity
    if submission_df.isnull().values.any():
        submission_df = submission_df.fillna("")
        
    # Save output
    os.makedirs(OUTPUT_DATA_PATH, exist_ok=True)
    submission_path = os.path.join(OUTPUT_DATA_PATH, "submission.csv")
    submission_df.to_csv(submission_path, index=False)
    print(f"Ensemble complete. Submission saved to {submission_path}")
    
    return submission_df