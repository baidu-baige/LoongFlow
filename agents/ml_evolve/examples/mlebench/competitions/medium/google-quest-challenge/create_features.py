# -*- coding: utf-8 -*-
"""
LLM Generated Code
"""

from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from transformers import DebertaV2Tokenizer

BASE_DATA_PATH = "/root/workspace/evolux_ml/output/mlebench/google-quest-challenge/prepared/public"
OUTPUT_DATA_PATH = "output/ea4f9f02-c6be-4ad0-a6fd-71930e8fb81e/15/executor/output"

# For type hinting, DT is assumed to be pandas DataFrame or Series
DT = pd.DataFrame | pd.Series


def create_features(
    X_train: DT,
    y_train: DT,
    X_val: DT,
    y_val: DT,
    X_test: DT
) -> Tuple[DT, DT, DT, DT, DT]:
    """
    Creates features for a single fold of cross-validation using DeBERTa-v3-large strategy.
    
    Transforms raw text and metadata into a high-density feature set tailored for the 
    DeBERTa-v3-large backbone, including structural features, interaction cues, and metadata.
    """
    # Initialize Tokenizer (microsoft/deberta-v3-large uses DebertaV2Tokenizer)
    tokenizer = DebertaV2Tokenizer.from_pretrained('microsoft/deberta-v3-large')

    # Consistency with DeBERTa-v3 special token IDs
    cls_id = 1
    sep_id = 2
    pad_id = 0

    def get_token_features(df, tokenizer):
        # Batch encode for efficiency
        t_all = tokenizer.batch_encode_plus(df['question_title'].astype(str).tolist(), add_special_tokens=False)[
            'input_ids']
        b_all = tokenizer.batch_encode_plus(df['question_body'].astype(str).tolist(), add_special_tokens=False)[
            'input_ids']
        a_all = tokenizer.batch_encode_plus(df['answer'].astype(str).tolist(), add_special_tokens=False)['input_ids']

        ids_list = []
        mask_list = []

        for t_raw, b_raw, a_raw in zip(t_all, b_all, a_all):
            # Truncation Strategy
            # question_title: Max 128 tokens
            t_tokens = t_raw[:128]

            # question_body: Max 190 tokens (Head 100 + Tail 90)
            if len(b_raw) > 190:
                b_tokens = b_raw[:100] + b_raw[-90:]
            else:
                b_tokens = b_raw

            # answer: Max 190 tokens (Head 100 + Tail 90)
            if len(a_raw) > 190:
                a_tokens = a_raw[:100] + a_raw[-90:]
            else:
                a_tokens = a_raw

            # Sequence Construction: [CLS] title [SEP] body [SEP] answer [SEP]
            # Max possible length: 1 + 128 + 1 + 190 + 1 + 190 + 1 = 512
            ids = [cls_id] + t_tokens + [sep_id] + b_tokens + [sep_id] + a_tokens + [sep_id]

            # Final truncation and Padding
            ids = ids[:512]
            mask = [1] * len(ids)

            pad_len = 512 - len(ids)
            ids += [pad_id] * pad_len
            mask += [0] * pad_len

            ids_list.append(ids)
            mask_list.append(mask)

        return np.array(ids_list, dtype=np.int32), np.array(mask_list, dtype=np.int32)

    def get_structural_features(df):
        out = pd.DataFrame(index=df.index)

        # Structural Features (Separated Q/A)
        out['q_code'] = df['question_body'].astype(str).str.count('```').fillna(0)
        out['a_code'] = df['answer'].astype(str).str.count('```').fillna(0)
        out['q_latex'] = df['question_body'].astype(str).str.count(r'\$').fillna(0)
        out['a_latex'] = df['answer'].astype(str).str.count(r'\$').fillna(0)
        out['q_link'] = df['question_body'].astype(str).str.count('http').fillna(0)
        out['a_link'] = df['answer'].astype(str).str.count('http').fillna(0)
        out['q_question'] = df['question_body'].astype(str).str.count(r'\?').fillna(0)

        # Interaction Features: Character-level Jaccard similarity (Question Title+Body vs Answer)
        def char_jaccard(row):
            q = str(row['question_title']) + " " + str(row['question_body'])
            a = str(row['answer'])
            set_q = set(q)
            set_a = set(a)
            intersection = len(set_q.intersection(set_a))
            union = len(set_q.union(set_a))
            return float(intersection) / union if union > 0 else 0.0

        out['jaccard_similarity'] = df.apply(char_jaccard, axis=1)

        # Lengths (as part of numeric feature set)
        out['q_len'] = df['question_body'].astype(str).str.len().fillna(0)
        out['a_len'] = df['answer'].astype(str).str.len().fillna(0)
        out['t_len'] = df['question_title'].astype(str).str.len().fillna(0)

        return out

    # Determine metadata mappings based on training data only
    top_hosts = X_train['host'].value_counts().index[:20].tolist()

    def get_meta(df, top_hosts):
        m = pd.DataFrame(index=df.index)
        m['host_clean'] = df['host'].apply(lambda x: x if x in top_hosts else 'other')
        m['category'] = df['category']
        return m

    # 1. Structural and Interaction Features
    train_struct = get_structural_features(X_train)
    val_struct = get_structural_features(X_val)
    test_struct = get_structural_features(X_test)

    struct_cols = train_struct.columns.tolist()
    scaler = StandardScaler()
    scaler.fit(train_struct)

    # 2. Metadata Features (One-hot encoding)
    train_meta = get_meta(X_train, top_hosts)
    val_meta = get_meta(X_val, top_hosts)
    test_meta = get_meta(X_test, top_hosts)

    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    ohe.fit(train_meta)
    ohe_cols = ohe.get_feature_names_out().tolist()

    # 3. Text Features (Tokenization)
    train_ids, train_mask = get_token_features(X_train, tokenizer)
    val_ids, val_mask = get_token_features(X_val, tokenizer)
    test_ids, test_mask = get_token_features(X_test, tokenizer)

    # Assembly helper function to combine all features
    def assemble(ids, mask, struct_df, meta_df, idx, scaler, ohe):
        # Create DataFrames for token IDs and masks
        df_ids = pd.DataFrame(ids, columns=[f'input_ids_{i}' for i in range(512)], index=idx)
        df_mask = pd.DataFrame(mask, columns=[f'attention_mask_{i}' for i in range(512)], index=idx)

        # Scale numeric features
        scaled_struct_vals = scaler.transform(struct_df)
        scaled_struct = pd.DataFrame(scaled_struct_vals, columns=struct_cols, index=idx)

        # Apply One-Hot Encoding
        ohe_vals = ohe.transform(meta_df)
        ohe_feats = pd.DataFrame(ohe_vals, columns=ohe_cols, index=idx)

        # Concatenate all components
        combined = pd.concat([df_ids, df_mask, scaled_struct, ohe_feats], axis=1)
        # Ensure no NaNs or Infs
        combined = combined.replace([np.inf, -np.inf], 0).fillna(0)
        return combined

    # Create final transformed feature sets
    X_train_transformed = assemble(train_ids, train_mask, train_struct, train_meta, X_train.index, scaler, ohe)
    X_val_transformed = assemble(val_ids, val_mask, val_struct, val_meta, X_val.index, scaler, ohe)
    X_test_transformed = assemble(test_ids, test_mask, test_struct, test_meta, X_test.index, scaler, ohe)

    return X_train_transformed, y_train, X_val_transformed, y_val, X_test_transformed
