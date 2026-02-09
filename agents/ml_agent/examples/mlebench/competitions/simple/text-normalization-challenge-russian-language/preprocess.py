import os
import cudf
import numpy as np
from typing import Tuple
from cuml.preprocessing import LabelEncoder

# Task-adaptive type definitions using RAPIDS cuDF for high-performance GPU processing.
# cuDF is essential for processing the large-scale Russian text normalization dataset (~10M tokens).
X = cudf.DataFrame
y = cudf.Series

BASE_DATA_PATH = "/mnt/pfs/loongflow/devmachine/h20-12/evolux/output/mlebench/text-normalization-challenge-russian-language/prepared/public"
OUTPUT_DATA_PATH = "output/ecfe1a48-59fb-4170-a38b-6ffb4a298ec0/10/executor/output"

def get_majority_vote(df: cudf.DataFrame, group_cols: list, target_col: str) -> cudf.DataFrame:
    """
    Computes the most frequent target value for each group using GPU acceleration.
    Essential for building robust normalization dictionaries.
    """
    # Count occurrences of each (group, target) pair
    counts = df.groupby(group_cols + [target_col]).size().reset_index(name='count')
    # Sort by group columns and count (descending) to bring majority to the top
    counts = counts.sort_values(by=group_cols + ['count'], ascending=[True] * len(group_cols) + [False])
    # Keep the first occurrence for each group, which is the majority vote
    majority = counts.drop_duplicates(subset=group_cols)
    return majority.drop(columns=['count'])

def preprocess(
    X_train: X,
    y_train: y,
    X_val: X,
    y_val: y,
    X_test: X
) -> Tuple[X, y, X, y, X]:
    """
    Transforms raw text data into model-ready format for the Russian Text Normalization task.
    Implements Contextual Windowing, Multi-Case Preposition Detection, and Phonetic/Vowel heuristics.
    """
    print("Preprocessing started...")
    os.makedirs(OUTPUT_DATA_PATH, exist_ok=True)

    # 1. Dictionary Building (Fit on X_train only to avoid data leakage)
    # These dictionaries map raw tokens to their most frequent normalized forms.
    print("Building normalization dictionaries...")
    norm_df = X_train[['class', 'before']].copy()
    norm_df['class'] = norm_df['class'].astype(str)
    norm_df['after'] = y_train.astype(str)

    # Global Dict: before -> most frequent after string
    get_majority_vote(norm_df, ['before'], 'after').to_parquet(
        os.path.join(OUTPUT_DATA_PATH, "global_dict.parquet")
    )
    # Class-Tag Dict: (class, before) -> most frequent after string
    get_majority_vote(norm_df, ['class', 'before'], 'after').to_parquet(
        os.path.join(OUTPUT_DATA_PATH, "class_tag_dict.parquet")
    )
    # Identity Dict: tokens that don't change during normalization
    identity_df = norm_df[norm_df['before'] == norm_df['after']]
    get_majority_vote(identity_df, ['before'], 'after').to_parquet(
        os.path.join(OUTPUT_DATA_PATH, "identity_dict.parquet")
    )

    # 2. Target Class Encoding
    # The pipeline stage 4 expects encoded semiotic class labels as the target for the classifier.
    print("Encoding target class labels...")
    le_class = LabelEncoder()
    y_train_processed = le_class.fit_transform(X_train['class'].astype(str))
    y_val_processed = le_class.transform(X_val['class'].astype(str))
    
    # Save mapping for inference-time class ID to name conversion
    cudf.DataFrame({
        'class_name': le_class.classes_,
        'class_id': np.arange(len(le_class.classes_), dtype='int32')
    }).to_parquet(os.path.join(OUTPUT_DATA_PATH, "class_mapping.parquet"))

    # 3. Feature Engineering
    # Define signals for Russian grammar (case triggers) and phonetic transliteration.
    vowel_regex = r'[аеёиоуыэюяaeiouy]'
    gen_preps = ['до', 'из', 'от', 'без', 'у', 'для', 'около', 'после']
    dat_preps = ['к', 'по']
    prep_preps = ['в', 'на', 'о', 'об']

    def extract_features(df: X) -> X:
        df = df.copy()
        df['before'] = df['before'].fillna('<PAD>')
        
        # Contextual Windowing (+/- 2 tokens)
        df['prev2_before'] = df.groupby('sentence_id')['before'].shift(2).fillna('<PAD>')
        df['prev1_before'] = df.groupby('sentence_id')['before'].shift(1).fillna('<PAD>')
        df['next1_before'] = df.groupby('sentence_id')['before'].shift(-1).fillna('<PAD>')
        df['next2_before'] = df.groupby('sentence_id')['before'].shift(-2).fillna('<PAD>')
        
        # Multi-Case Preposition Detection (Genitive, Dative, Prepositional)
        # These features help predict inflected forms of numerals and dates.
        df['is_gen_prep'] = (df['prev1_before'].str.lower().isin(gen_preps) | 
                             df['prev2_before'].str.lower().isin(gen_preps)).astype('int8')
        df['is_dat_prep'] = (df['prev1_before'].str.lower().isin(dat_preps) | 
                             df['prev2_before'].str.lower().isin(dat_preps)).astype('int8')
        df['is_prep_prep'] = (df['prev1_before'].str.lower().isin(prep_preps) | 
                              df['prev2_before'].str.lower().isin(prep_preps)).astype('int8')
        
        # Phonetic (Latin) and Morphological (Vowel) signals
        # Vowel counts help with gender/case agreement heuristics.
        df['v_cnt'] = df['before'].str.lower().str.count(vowel_regex).fillna(0).astype('int16')
        df['h_lat'] = df['before'].str.contains(r'[a-zA-Z]').fillna(False).astype('int8')
        
        # Shift signals to capture neighbor vowel/phonetic context
        for i, suffix in [(1, 'p1'), (2, 'p2')]:
            df[f'v_cnt_{suffix}'] = df.groupby('sentence_id')['v_cnt'].shift(i).fillna(0).astype('int16')
            df[f'h_lat_{suffix}'] = df.groupby('sentence_id')['h_lat'].shift(i).fillna(0).astype('int8')
        for i, suffix in [(-1, 'n1'), (-2, 'n2')]:
            df[f'v_cnt_{suffix}'] = df.groupby('sentence_id')['v_cnt'].shift(i).fillna(0).astype('int16')
            df[f'h_lat_{suffix}'] = df.groupby('sentence_id')['h_lat'].shift(i).fillna(0).astype('int8')

        # Basic Orthographic and Semiotic Regex features
        df['len'] = df['before'].str.len().fillna(0).astype('int32')
        df['is_upper'] = df['before'].str.isupper().fillna(False).astype('int8')
        df['is_digit'] = df['before'].str.match(r'^\d+$').fillna(False).astype('int8')
        df['is_date'] = df['before'].str.contains(r'\d{1,4}[-./]\d{1,2}[-./]\d{1,4}').fillna(False).astype('int8')
        df['is_time'] = df['before'].str.contains(r'\d{1,2}:\d{2}').fillna(False).astype('int8')
        df['is_electronic'] = df['before'].str.contains(r'http|www|\.com|\.ru|@').fillna(False).astype('int8')
        
        return df

    print("Extracting contextual and phonetic features...")
    X_train_ext = extract_features(X_train)
    X_val_ext = extract_features(X_val)
    X_test_ext = extract_features(X_test)

    # 4. Token Encoding
    # Numerical representation of strings for the machine learning model.
    print("Encoding tokens into numerical indices...")
    # Calculate vocab from training strings only
    vocab = cudf.concat([X_train_ext['before'], cudf.Series(['<PAD>'])]).unique()
    token_map = cudf.Series(np.arange(1, len(vocab) + 1, dtype='int32'), index=vocab)
    
    def apply_mapping(df, mapping):
        token_cols = ['before', 'prev1_before', 'prev2_before', 'next1_before', 'next2_before']
        for col in token_cols:
            df[col] = df[col].map(mapping).fillna(0).astype('int32')
        return df

    X_train_proc = apply_mapping(X_train_ext, token_map)
    X_val_proc = apply_mapping(X_val_ext, token_map)
    X_test_proc = apply_mapping(X_test_ext, token_map)

    # 5. Finalize Column Structure
    # Ensure identical feature sets across all splits.
    feature_cols = [
        'sentence_id', 'token_id', 'before', 'prev1_before', 'prev2_before', 
        'next1_before', 'next2_before', 'len', 'is_upper', 'is_digit', 
        'is_date', 'is_time', 'is_electronic', 'is_gen_prep', 'is_dat_prep', 
        'is_prep_prep', 'v_cnt', 'v_cnt_p1', 'v_cnt_p2', 'v_cnt_n1', 'v_cnt_n2',
        'h_lat', 'h_lat_p1', 'h_lat_p2', 'h_lat_n1', 'h_lat_n2'
    ]
    
    X_train_proc = X_train_proc[feature_cols]
    X_val_proc = X_val_proc[feature_cols]
    X_test_proc = X_test_proc[feature_cols]

    # Verify data integrity
    for name, df in [("train", X_train_proc), ("val", X_val_proc), ("test", X_test_proc)]:
        if df.isna().any().any():
            raise ValueError(f"NaN values detected in processed {name} data.")

    print(f"Preprocessing complete. Total features: {len(feature_cols)}")
    return X_train_proc, y_train_processed, X_val_proc, y_val_processed, X_test_proc