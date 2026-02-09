import pandas as pd
import numpy as np
import os
import spacy
import nltk
from typing import Tuple, Any
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, chi2
from scipy.sparse import hstack, csr_matrix
from transformers import AutoTokenizer
from joblib import Parallel, delayed

# Task-adaptive type definitions
X = pd.DataFrame
y = np.ndarray

def preprocess(
    X_train: X,
    y_train: pd.Series,
    X_val: X,
    y_val: pd.Series,
    X_test: X
) -> Tuple[X, y, X, y, X]:
    """
    Transforms raw text into a high-dimensional multi-modal representation:
    1. Multi-Transformer Tokenization (DeBERTa-v3-large & RoBERTa-large).
    2. Pruned TF-IDF (Word 1-3g & Char 2-6g) using Chi-square selection.
    3. Deep syntactic and stylistic meta-features (spaCy-based).
    """
    print("Beginning preprocessing: Multi-modal feature extraction and syntactic depth analysis.")
    
    # Disable parallelism in tokenizers to avoid deadlocks
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # 1. Target Encoding
    print("Encoding author labels...")
    label_encoder = LabelEncoder()
    y_train_processed = label_encoder.fit_transform(y_train)
    y_val_processed = label_encoder.transform(y_val)

    # 2. Syntactic & Stylistic Feature Extraction
    print("Loading spaCy/NLTK resources...")
    try:
        nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])
    except OSError:
        os.system("python -m spacy download en_core_web_sm")
        nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])

    def count_syllables(text: str) -> int:
        words = text.lower().split()
        total = 0
        vowels = "aeiouy"
        for word in words:
            word = word.strip(".:;?!\"()'")
            if not word: continue
            count = 0
            if word[0] in vowels: count += 1
            for i in range(1, len(word)):
                if word[i] in vowels and word[i-1] not in vowels: count += 1
            if word.endswith("e"): count -= 1
            total += max(1, count)
        return total

    def extract_dense_features(df: pd.DataFrame) -> np.ndarray:
        texts = df['text'].tolist()
        batch_size = 256
        n_process = 36 # CPU Cores
        
        puncts = ['.', ',', ';', ':', '!', '?', '"', "'"]
        meta_results = []

        print(f"Processing syntactic/POS features with spaCy (n_process={n_process})...")
        for doc in nlp.pipe(texts, batch_size=batch_size, n_process=n_process):
            word_count = len(doc)
            div = max(1, word_count)
            text = doc.text
            
            feat = {}
            # Punctuation (8 features)
            for p in puncts:
                feat[f'punct_{p}'] = text.count(p)
            
            # POS Densities (4 features: JJ, RB, VB, NN)
            feat['jj_density'] = sum(1 for t in doc if t.tag_.startswith('JJ')) / div
            feat['rb_density'] = sum(1 for t in doc if t.tag_.startswith('RB')) / div
            feat['vb_density'] = sum(1 for t in doc if t.tag_.startswith('VB')) / div
            feat['nn_density'] = sum(1 for t in doc if t.tag_.startswith('NN')) / div
            
            # Syntactic Complexity (2 features)
            dep_dist = sum(abs(t.i - t.head.i) for t in doc)
            feat['avg_dep_dist'] = dep_dist / div
            
            def get_depth(t):
                d = 0
                curr = t
                while curr != curr.head:
                    d += 1
                    curr = curr.head
                return d
            feat['max_depth'] = max([get_depth(t) for t in doc], default=0)
            
            # Stylistic Stats (11 features)
            feat['char_len'] = len(text)
            feat['word_count'] = word_count
            feat['avg_word_len'] = len(text) / div
            feat['cap_count'] = sum(1 for t in doc if t.text.isupper())
            feat['stop_count'] = sum(1 for t in doc if t.is_stop)
            feat['non_stop_count'] = word_count - feat['stop_count']
            feat['upper_word_count'] = sum(1 for t in doc if t.text.isupper() and len(t.text) > 1)
            feat['unique_word_count'] = len(set(t.text.lower() for t in doc))
            feat['title_case_count'] = sum(1 for t in doc if t.text.istitle())
            feat['ttr'] = feat['unique_word_count'] / div
            
            syllables = count_syllables(text)
            feat['readability_proxy'] = word_count + (syllables / div)
            
            meta_results.append(list(feat.values()))
            
        return np.array(meta_results, dtype=np.float32)

    print("Extracting dense features for Train, Val, and Test...")
    meta_train = extract_dense_features(X_train)
    meta_val = extract_dense_features(X_val)
    meta_test = extract_dense_features(X_test)

    # Scaling dense features
    scaler = StandardScaler()
    meta_train = scaler.fit_transform(meta_train)
    meta_val = scaler.transform(meta_val)
    meta_test = scaler.transform(meta_test)

    # 3. Pruned TF-IDF (Word 1-3g, Char 2-6g)
    print("Computing TF-IDF and pruning with Chi-square (k=40000)...")
    token_pattern = r'\w+|[^\w\s]'
    
    word_vec = TfidfVectorizer(
        ngram_range=(1, 3), min_df=3, sublinear_tf=True, 
        lowercase=False, token_pattern=token_pattern
    )
    char_vec = TfidfVectorizer(
        ngram_range=(2, 6), min_df=3, sublinear_tf=True, 
        lowercase=False, analyzer='char'
    )

    tfidf_train_raw = hstack([
        word_vec.fit_transform(X_train['text']),
        char_vec.fit_transform(X_train['text'])
    ], format='csr')
    
    tfidf_val_raw = hstack([
        word_vec.transform(X_val['text']),
        char_vec.transform(X_val['text'])
    ], format='csr')
    
    tfidf_test_raw = hstack([
        word_vec.transform(X_test['text']),
        char_vec.transform(X_test['text'])
    ], format='csr')

    k_features = min(40000, tfidf_train_raw.shape[1])
    selector = SelectKBest(chi2, k=k_features)
    tfidf_train = selector.fit_transform(tfidf_train_raw, y_train_processed)
    tfidf_val = selector.transform(tfidf_val_raw)
    tfidf_test = selector.transform(tfidf_test_raw)

    # 4. Multi-Transformer Tokenization
    def get_tokenization(model_name: str, df: pd.DataFrame):
        print(f"Tokenizing with {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        enc = tokenizer(
            df['text'].tolist(),
            max_length=320,
            padding='max_length',
            truncation=True,
            return_tensors='np'
        )
        return enc['input_ids'], enc['attention_mask']

    # DeBERTa-v3-large
    deb_train_ids, deb_train_mask = get_tokenization("microsoft/deberta-v3-large", X_train)
    deb_val_ids, deb_val_mask = get_tokenization("microsoft/deberta-v3-large", X_val)
    deb_test_ids, deb_test_mask = get_tokenization("microsoft/deberta-v3-large", X_test)

    # RoBERTa-large
    rob_train_ids, rob_train_mask = get_tokenization("roberta-large", X_train)
    rob_val_ids, rob_val_mask = get_tokenization("roberta-large", X_val)
    rob_test_ids, rob_test_mask = get_tokenization("roberta-large", X_test)

    # 5. Packaging Results
    def package(ids_deb, mask_deb, ids_rob, mask_rob, tfidf, dense):
        return pd.DataFrame({
            'deb_input_ids': list(ids_deb),
            'deb_attention_mask': list(mask_deb),
            'rob_input_ids': list(ids_rob),
            'rob_attention_mask': list(mask_rob),
            'tfidf_features': [tfidf[i] for i in range(tfidf.shape[0])],
            'dense_meta': list(dense)
        })

    X_train_processed = package(deb_train_ids, deb_train_mask, rob_train_ids, rob_train_mask, tfidf_train, meta_train)
    X_val_processed = package(deb_val_ids, deb_val_mask, rob_val_ids, rob_val_mask, tfidf_val, meta_val)
    X_test_processed = package(deb_test_ids, deb_test_mask, rob_test_ids, rob_test_mask, tfidf_test, meta_test)

    # Validate output integrity
    for df_proc in [X_train_processed, X_val_processed, X_test_processed]:
        if df_proc.isnull().values.any():
            raise ValueError("NaN values detected in processed features.")

    print(f"Preprocessing complete. Dense features: {meta_train.shape[1]}, TF-IDF dimension: {tfidf_train.shape[1]}")
    return X_train_processed, y_train_processed, X_val_processed, y_val_processed, X_test_processed