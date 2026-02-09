import pandas as pd
import numpy as np
import torch
import re
import nltk
from transformers import AutoTokenizer, AutoModel
from sklearn.decomposition import PCA
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from collections import Counter
from typing import Tuple, Any

# Task-adaptive type definitions
X = pd.DataFrame
y = pd.Series

# Ensure NLTK resources are available
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon', quiet=True)

# Readability Fallback
def get_readability_score(text: str) -> float:
    """Calculates simplified Flesch-Kincaid Grade Level."""
    words = text.split()
    if not words: return 0.0
    sentences = max(1, text.count('.') + text.count('!') + text.count('?'))
    syllables = 0
    for word in words:
        word = word.lower()
        count = len(re.findall(r'[aeiouy]', word))
        syllables += max(1, count)
    return 0.39 * (len(words)/sentences) + 11.8 * (syllables/len(words)) - 15.59

def get_sbert_embeddings(texts: pd.Series, model: Any, tokenizer: Any, device: torch.device) -> np.ndarray:
    """Generates SBERT embeddings for a series of texts."""
    all_embeddings = []
    text_list = texts.astype(str).tolist()
    batch_size = 64
    for i in range(0, len(text_list), batch_size):
        batch = text_list[i:i+batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        # Mean Pooling
        mask = inputs['attention_mask'].unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
        embeddings = torch.sum(outputs.last_hidden_state * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)
        # Normalize
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        all_embeddings.append(embeddings.cpu().numpy())
    return np.vstack(all_embeddings)

def preprocess(
    X_train: X,
    y_train: y,
    X_val: X,
    y_val: y,
    X_test: X
) -> Tuple[X, y, X, y, X]:
    """
    Multimodal feature extractor capturing semantic intent, social capital, and community-specific signals.
    """
    print("Preprocess: Initializing Multimodal Extraction Pipeline")
    
    # 0. Setup Metadata & Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = 'sentence-transformers/all-MiniLM-L6-v2'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()
    
    vader = SentimentIntensityAnalyzer()
    hardship_keywords = ['rent', 'bills', 'money', 'hungry', 'job', 'broke', 'struggling', 'evicted', 'need', 'food', 'help', 'lost', 'struggle', 'low', 'short']
    
    # 1. Fit Global Transformers on Train
    print("Fitting transformers on Training data...")
    
    # Text Preparation
    def get_combined_text(df):
        return (df['request_title'].astype(str) + " " + df['request_text_edit_aware'].astype(str)).fillna("")

    train_texts = get_combined_text(X_train)
    val_texts = get_combined_text(X_val)
    test_texts = get_combined_text(X_test)

    # SBERT + PCA (100)
    train_embs = get_sbert_embeddings(train_texts, model, tokenizer, device)
    pca = PCA(n_components=100, random_state=42)
    train_pca = pca.fit_transform(train_embs)
    
    # Subreddits (Top 100 Binary)
    all_train_subs = [sub for sublist in X_train['requester_subreddits_at_request'] for sub in sublist]
    top_100_subs = [s for s, c in Counter(all_train_subs).most_common(100)]
    mlb = MultiLabelBinarizer(classes=top_100_subs)
    mlb.fit(X_train['requester_subreddits_at_request'])
    
    # Subreddits (Bayesian TE stats)
    alpha = 25.0
    global_mean = y_train.mean()
    sub_map = {}
    for subs, label in zip(X_train['requester_subreddits_at_request'], y_train):
        for s in subs:
            if s not in sub_map:
                sub_map[s] = {'sum': 0.0, 'count': 0}
            sub_map[s]['sum'] += float(label)
            sub_map[s]['count'] += 1

    def get_te_score(subs, label=None):
        if not subs: return global_mean
        scores = []
        for s in subs:
            if s in sub_map:
                s_sum, s_count = sub_map[s]['sum'], sub_map[s]['count']
                if label is not None: # Leave-One-Out for Train
                    s_sum -= float(label)
                    s_count -= 1
                scores.append((s_sum + alpha * global_mean) / (s_count + alpha))
            else:
                scores.append(global_mean)
        return np.mean(scores)

    # 2. Feature Extraction Function
    def extract_all_features(df: X, pca_data: np.ndarray, y_labels: y = None) -> X:
        indices = df.index
        combined_text = get_combined_text(df)
        
        # PCA SBERT Features
        feat_df = pd.DataFrame(pca_data, columns=[f'pca_sbert_{i}' for i in range(100)], index=indices)
        
        # Temporal
        ts_utc = pd.to_datetime(df['unix_timestamp_of_request_utc'], unit='s')
        hour_sin = np.sin(2 * np.pi * ts_utc.dt.hour / 24)
        feat_df['hour_sin'] = hour_sin
        feat_df['hour_cos'] = np.cos(2 * np.pi * ts_utc.dt.hour / 24)
        
        # Linguistic (Pronouns)
        def count_pronouns(text):
            text = text.lower()
            sing = len(re.findall(r'\b(i|me|my|mine|myself)\b', text))
            plur = len(re.findall(r'\b(we|us|our|ours|ourselves)\b', text))
            words = max(1, len(text.split()))
            return sing, plur, sing/words, plur/words
        
        pronoun_res = combined_text.apply(count_pronouns)
        feat_df['1p_sing_cnt'] = pronoun_res.apply(lambda x: x[0])
        feat_df['1p_plur_cnt'] = pronoun_res.apply(lambda x: x[1])
        feat_df['1p_sing_ratio'] = pronoun_res.apply(lambda x: x[2])
        feat_df['1p_plur_ratio'] = pronoun_res.apply(lambda x: x[3])
        
        # Sentiment & Readability
        senti = combined_text.apply(lambda x: vader.polarity_scores(x)['compound'])
        feat_df['vader_compound'] = senti
        readability = combined_text.apply(get_readability_score)
        feat_df['readability'] = readability
        
        # Core & Keywords
        feat_df['hardship_count'] = combined_text.apply(lambda x: sum(1 for w in hardship_keywords if re.search(f'\\b{w}\\b', x.lower())))
        feat_df['is_giver_known'] = (df['giver_username_if_known'] != 'N/A').astype(int) if 'giver_username_if_known' in df.columns else 0
        feat_df['punct_density'] = combined_text.apply(lambda x: len(re.findall(r'[^\w\s]', x)) / (len(x) + 1))
        
        # Reputation/Activity
        log_age = np.log1p(df['requester_account_age_in_days_at_request'].clip(lower=0))
        log_karma = np.log1p(df['requester_upvotes_plus_downvotes_at_request'].clip(lower=0))
        feat_df['log_age'] = log_age
        feat_df['log_karma'] = log_karma
        
        # Metadata Activity (Log1p scaled)
        activity_cols = [
            'requester_days_since_first_post_on_raop_at_request',
            'requester_number_of_comments_at_request',
            'requester_number_of_comments_in_raop_at_request',
            'requester_number_of_posts_at_request',
            'requester_number_of_posts_on_raop_at_request',
            'requester_number_of_subreddits_at_request',
            'requester_upvotes_minus_downvotes_at_request'
        ]
        for col in activity_cols:
            feat_df[f'log_{col}'] = np.log1p(df[col].clip(lower=0)) if col in df.columns else 0
            
        # Recursive & Contextual Interactions
        feat_df['inter_3way_recursive'] = (hour_sin * feat_df['hardship_count']) * log_age
        feat_df['inter_vader_age'] = senti * log_age
        feat_df['inter_read_karma'] = readability * log_karma
        
        # Subreddits Binary
        sub_bin = mlb.transform(df['requester_subreddits_at_request'])
        sub_bin_df = pd.DataFrame(sub_bin, columns=[f'sub_top_{i}' for i in range(100)], index=indices)
        feat_df = pd.concat([feat_df, sub_bin_df], axis=1)
        
        # Subreddits Bayesian TE
        if y_labels is not None:
            feat_df['sub_te'] = [get_te_score(s, l) for s, l in zip(df['requester_subreddits_at_request'], y_labels)]
        else:
            feat_df['sub_te'] = df['requester_subreddits_at_request'].apply(lambda x: get_te_score(x))
            
        return feat_df.fillna(0).replace([np.inf, -np.inf], 0)

    # 3. Apply Extraction
    print("Extracting features for all splits...")
    X_train_f = extract_all_features(X_train, train_pca, y_train)
    
    val_embs = get_sbert_embeddings(val_texts, model, tokenizer, device)
    val_pca = pca.transform(val_embs)
    X_val_f = extract_all_features(X_val, val_pca)
    
    test_embs = get_sbert_embeddings(test_texts, model, tokenizer, device)
    test_pca = pca.transform(test_embs)
    X_test_f = extract_all_features(X_test, test_pca)

    # 4. Final Scaling and Packaging
    print(f"Standardizing {X_train_f.shape[1]} numeric features...")
    scaler = StandardScaler()
    scaler.fit(X_train_f)
    
    def finalize_df(f_df, raw_texts):
        scaled_data = scaler.transform(f_df)
        final_df = pd.DataFrame(scaled_data, columns=f_df.columns, index=f_df.index)
        final_df['raw_text'] = raw_texts.values
        return final_df

    X_train_processed = finalize_df(X_train_f, train_texts)
    X_val_processed = finalize_df(X_val_f, val_texts)
    X_test_processed = finalize_df(X_test_f, test_texts)
    
    y_train_processed = y_train.astype(int)
    y_val_processed = y_val.astype(int)

    print(f"Preprocessing complete. Total features: {X_train_processed.shape[1]}")
    return X_train_processed, y_train_processed, X_val_processed, y_val_processed, X_test_processed