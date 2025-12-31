# -*- coding: utf-8 -*-
"""
LLM Generated Code
"""

from typing import Tuple

import cudf
import numpy as np
import pandas as pd

BASE_DATA_PATH = "/root/workspace/evolux_ml/output/mlebench/text-normalization-challenge-english-language/prepared/public"
OUTPUT_DATA_PATH = "output/12e29d80-a70c-426a-9331-de3aa1a6ce7c/14/executor/output"

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
    Creates high-dimensional features for text normalization using GPU acceleration.
    """
    # 1. Prepare data for unified processing on GPU
    X_train_c = X_train.copy()
    X_val_c = X_val.copy()
    X_test_c = X_test.copy()

    # Assign markers to split back later
    X_train_c['_set_'] = 0
    X_val_c['_set_'] = 1
    X_test_c['_set_'] = 2

    # Ensure sentence_id is unique across sets to avoid context bleed during shifts
    val_sid_offset = int(X_train_c['sentence_id'].max()) + 1 if len(X_train_c) > 0 else 0
    X_val_c['sentence_id'] += val_sid_offset

    test_sid_offset = int(X_val_c['sentence_id'].max()) + 1 if len(X_val_c) > 0 else 0
    X_test_c['sentence_id'] += test_sid_offset

    # Move to GPU
    gdf = cudf.concat([
        cudf.from_pandas(X_train_c),
        cudf.from_pandas(X_val_c),
        cudf.from_pandas(X_test_c)
    ])

    # Ensure 'before' is string type and handle NaNs
    gdf['before_str'] = gdf['before'].astype(str).fillna("")

    # 2. Structural Flags
    gdf['len'] = gdf['before_str'].str.len().fillna(0).astype('int32')
    gdf['digit_count'] = gdf['before_str'].str.count(r'[0-9]').fillna(0).astype('int32')
    gdf['alpha_count'] = gdf['before_str'].str.count(r'[a-zA-Z]').fillna(0).astype('int32')
    gdf['symbol_count'] = (gdf['len'] - gdf['digit_count'] - gdf['alpha_count']).clip(0).astype('int32')

    gdf['is_caps'] = gdf['before_str'].str.isupper().fillna(False).astype('uint8')
    gdf['is_title'] = gdf['before_str'].str.istitle().fillna(False).astype('uint8')
    gdf['has_hyphen'] = gdf['before_str'].str.contains(r'-').fillna(False).astype('uint8')
    gdf['has_dot'] = gdf['before_str'].str.contains(r'\.').fillna(False).astype('uint8')
    gdf['is_single_char'] = (gdf['len'] == 1).astype('uint8')

    # 3. Regex Intent Features (Expanded)
    # is_abbrev: ^[A-Z][a-z]{1,2}\.$
    gdf['is_abbrev'] = gdf['before_str'].str.match(r'^[A-Z][a-z]{1,2}\.$').fillna(False).astype('uint8')
    # is_fraction: \d+/\d+
    gdf['is_fraction'] = gdf['before_str'].str.contains(r'\d+/\d+').fillna(False).astype('uint8')
    # has_measure_unit: Digits followed by ft, lb, km, %, oz, mph, in, mg, ml
    gdf['has_measure_unit'] = gdf['before_str'].str.contains(r'\d+(ft|lb|km|%|oz|mph|in|mg|ml)').fillna(False).astype(
        'uint8')
    # structural intents
    gdf['has_slash'] = gdf['before_str'].str.contains(r'/').fillna(False).astype('uint8')
    gdf['has_colon'] = gdf['before_str'].str.contains(r':').fillna(False).astype('uint8')
    gdf['has_currency'] = gdf['before_str'].str.contains(r'[$£€¥]').fillna(False).astype('uint8')
    # is_url: www, .com, .org
    gdf['is_url'] = gdf['before_str'].str.contains(r'www|\.com|\.org').fillna(False).astype('uint8')

    # 4. Positional Features
    max_tid = gdf.groupby('sentence_id')['token_id'].transform('max')
    gdf['rel_pos'] = (gdf['token_id'].astype('float32') / max_tid.clip(1)).astype('float32')

    # 5. Context Window Features (Size 3)
    # Context does not bleed across sentences by masking with sentence_id
    for i in [-3, -2, -1, 1, 2, 3]:
        shifted_sid = gdf['sentence_id'].shift(i)
        mask = (gdf['sentence_id'] == shifted_sid)
        gdf[f'before_{i}'] = gdf['before_str'].shift(i).where(mask, "")

    # 6. Character N-grams
    for n in [1, 2, 3]:
        gdf[f'pre_{n}'] = gdf['before_str'].str.slice(0, n)
        gdf[f'suf_{n}'] = gdf['before_str'].str.slice(-n)

    # 7. Categorical Encoding
    train_slice = gdf[gdf['_set_'] == 0]

    # Top 50,000 tokens for the primary target token and context window
    top_tokens = train_slice['before_str'].value_counts().head(50000).index.to_arrow().to_pylist()

    # Encode 'before' and all context windows using the same vocabulary
    gdf['before'] = gdf['before_str'].astype('category').cat.set_categories(top_tokens).cat.codes.astype(
        'int32').fillna(-1)
    for i in [-3, -2, -1, 1, 2, 3]:
        col = f'before_{i}'
        gdf[col] = gdf[col].astype('category').cat.set_categories(top_tokens).cat.codes.astype('int32').fillna(-1)

    # Encode Character N-grams (top 5000 for each prefix/suffix length)
    for n in [1, 2, 3]:
        for part in ['pre', 'suf']:
            col = f'{part}_{n}'
            top_ngrams = train_slice[col].value_counts().head(5000).index.to_arrow().to_pylist()
            gdf[col] = gdf[col].astype('category').cat.set_categories(top_ngrams).cat.codes.astype('int32').fillna(-1)

    # Clean up temporary string column
    gdf = gdf.drop(columns=['before_str'])

    # 8. Split back and convert to Pandas
    res_train = gdf[gdf['_set_'] == 0].drop(columns=['_set_']).to_pandas()
    res_val = gdf[gdf['_set_'] == 1].drop(columns=['_set_']).to_pandas()
    res_test = gdf[gdf['_set_'] == 2].drop(columns=['_set_']).to_pandas()

    # Restore original sentence_id values
    res_train['sentence_id'] = X_train['sentence_id'].values
    res_val['sentence_id'] = X_val['sentence_id'].values
    res_test['sentence_id'] = X_test['sentence_id'].values

    # Ensure consistent column ordering across sets
    cols = res_train.columns.tolist()
    res_val = res_val[cols]
    res_test = res_test[cols]

    # Ensure no NaN or Infinity values
    res_train = res_train.replace([np.inf, -np.inf], 0).fillna(0)
    res_val = res_val.replace([np.inf, -np.inf], 0).fillna(0)
    res_test = res_test.replace([np.inf, -np.inf], 0).fillna(0)

    return res_train, y_train, res_val, y_val, res_test
