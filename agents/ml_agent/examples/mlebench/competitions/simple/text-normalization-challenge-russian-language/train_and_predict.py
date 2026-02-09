import os
import cudf
import numpy as np
import pandas as pd
import xgboost as xgb
import re
from typing import Tuple, Dict, Any, Callable

# Dataset and Output Paths
BASE_DATA_PATH = "/mnt/pfs/loongflow/devmachine/h20-12/evolux/output/mlebench/text-normalization-challenge-russian-language/prepared/public"
OUTPUT_DATA_PATH = "output/ecfe1a48-59fb-4170-a38b-6ffb4a298ec0/10/executor/output"

# Task-adaptive type definitions
X = cudf.DataFrame
y = cudf.Series
Predictions = pd.Series

# Model Function Type Definition
ModelFn = Callable[
    [X, y, X, y, X],
    Tuple[Predictions, Predictions]
]

# ===== Training Functions =====

def train_phonetic_xgb_engine(
    X_train: X,
    y_train: y,
    X_val: X,
    y_val: y,
    X_test: X
) -> Tuple[Predictions, Predictions]:
    """
    Trains an XGBoost classifier for token categorization and applies a 
    phonetically-precise transformation engine for Russian text normalization.
    """
    print("Starting Training: XGBoost Phonetic-Aware Engine")
    
    # 1. Prepare Data for XGBoost
    drop_cols = ['sentence_id', 'token_id']
    train_feats = X_train.drop(columns=drop_cols)
    val_feats = X_val.drop(columns=drop_cols)
    test_feats = X_test.drop(columns=drop_cols)
    
    num_classes = int(y_train.max() + 1)
    
    # XGBoost Parameters optimized for GPU
    xgb_params = {
        'objective': 'multi:softmax',
        'num_class': num_classes,
        'tree_method': 'hist',
        'device': 'cuda',
        'n_estimators': 600,
        'learning_rate': 0.08,
        'max_depth': 8,
        'subsample': 0.85,
        'colsample_bytree': 0.85,
        'random_state': 42,
        'verbosity': 1,
        'n_jobs': 36,
        'early_stopping_rounds': 25
    }

    print(f"Training XGBoost on {train_feats.shape[0]} tokens...")
    model = xgb.XGBClassifier(**xgb_params)
    
    model.fit(
        train_feats, 
        y_train,
        eval_set=[(val_feats, y_val)],
        verbose=100
    )

    print("Generating class predictions...")
    val_class_preds = model.predict(val_feats)
    test_class_preds = model.predict(test_feats)

    # 2. Load Metadata and Dictionaries
    print("Loading normalization metadata...")
    class_mapping = pd.read_parquet(os.path.join(OUTPUT_DATA_PATH, "class_mapping.parquet"))
    id_to_class = class_mapping.set_index('class_id')['class_name'].to_dict()
    
    class_tag_dict = pd.read_parquet(os.path.join(OUTPUT_DATA_PATH, "class_tag_dict.parquet"))
    class_tag_lookup = class_tag_dict.set_index(['class', 'before'])['after'].to_dict()
    
    global_dict = pd.read_parquet(os.path.join(OUTPUT_DATA_PATH, "global_dict.parquet"))
    global_lookup = global_dict.set_index('before')['after'].to_dict()

    # 3. Reload original 'before' strings
    print("Extracting original strings for rule-based engine...")
    train_csv = os.path.join(BASE_DATA_PATH, "prepared_optimized", "ru_train.csv")
    test_csv = os.path.join(BASE_DATA_PATH, "prepared_optimized", "ru_test_2.csv")
    
    # Use pandas for high-concurrency string manipulation
    train_before = cudf.read_csv(train_csv, usecols=['before'], dtype={'before': 'string'})['before']
    test_before = cudf.read_csv(test_csv, usecols=['before'], dtype={'before': 'string'})['before']
    
    val_strings = train_before.iloc[X_val.index].to_pandas().values
    test_strings = test_before.to_pandas().values

    # 4. Russian Phonetic & Grammatical Engine

    def get_plural(n: int, forms: list) -> str:
        n = abs(int(n))
        if n % 10 == 1 and n % 100 != 11: return forms[0]
        elif 2 <= n % 10 <= 4 and (n % 100 < 10 or n % 100 >= 20): return forms[1]
        else: return forms[2]

    def get_plural_adj(n: int, forms: list) -> str:
        n = abs(int(n))
        if n % 10 == 1 and n % 100 != 11: return forms[0]
        return forms[1]

    numbers_nom = {
        0: 'ноль', 1: 'один', 2: 'два', 3: 'три', 4: 'четыре', 5: 'пять', 6: 'шесть', 7: 'семь', 8: 'восемь', 9: 'девять',
        10: 'десять', 11: 'одиннадцать', 12: 'двенадцать', 13: 'тринадцать', 14: 'четырнадцать', 15: 'пятнадцать', 16: 'шестнадцать', 17: 'семнадцать', 18: 'восемнадцать', 19: 'девятнадцать',
        20: 'двадцать', 30: 'тридцать', 40: 'сорок', 50: 'пятьдесят', 60: 'шестьдесят', 70: 'семьдесят', 80: 'восемьдесят', 90: 'девяносто',
        100: 'сто', 200: 'двести', 300: 'триста', 400: 'четыреста', 500: 'пятьсот', 600: 'шестьсот', 700: 'семьсот', 800: 'восемьсот', 900: 'девятьсот'
    }

    numbers_gen = {k: v.replace('ь', 'и').replace('я', 'и') if k >= 5 and k <= 30 else v for k, v in numbers_nom.items()}
    numbers_gen.update({1: 'одного', 2: 'двух', 3: 'трех', 4: 'четырех', 40: 'сорока', 90: 'девяноста', 100: 'ста'})

    ordinals_gen = {
        1: 'первого', 2: 'второго', 3: 'третьего', 4: 'четвертого', 5: 'пятого', 6: 'шестого', 7: 'седьмого', 8: 'восьмого', 9: 'девятого',
        10: 'десятого', 11: 'одиннадцатого', 12: 'двенадцатого', 13: 'тринадцатого', 14: 'четырнадцатого', 15: 'пятнадцатого', 16: 'шестнадцатого', 17: 'семнадцатого', 18: 'восемнадцатого', 19: 'девятнадцатого',
        20: 'двадцатого', 30: 'тридцатого', 40: 'сорокового', 50: 'пятидесятого', 60: 'шестидесятого', 70: 'семидесятого', 80: 'восьмидесятого', 90: 'девяностого', 100: 'сотого'
    }

    def num_to_ru(n, gender='m', ordinal=False, case='nom'):
        if n == 0: return 'ноль' if not ordinal else 'нулевого'
        parts = []
        rem = n
        if rem >= 1000:
            th = rem // 1000
            parts.append(num_to_ru(th, gender='f', case=case))
            parts.append(get_plural(th, ['тысяча', 'тысячи', 'тысяч']))
            rem %= 1000
        if rem >= 100:
            h = (rem // 100) * 100
            parts.append(numbers_nom[h])
            rem %= 100
        if rem > 0:
            if ordinal:
                if rem in ordinals_gen: parts.append(ordinals_gen[rem])
                else: 
                    parts.append(numbers_nom[(rem // 10) * 10])
                    parts.append(ordinals_gen[rem % 10])
            else:
                if rem in numbers_nom:
                    word = numbers_nom[rem]
                    if rem == 1: word = {'m': 'один', 'f': 'одна', 'n': 'одно'}[gender]
                    elif rem == 2: word = {'m': 'два', 'f': 'две', 'n': 'два'}[gender]
                    parts.append(word)
                else:
                    parts.append(numbers_nom[(rem // 10) * 10])
                    o = rem % 10
                    word = numbers_nom[o]
                    if o == 1: word = {'m': 'один', 'f': 'одна', 'n': 'одно'}[gender]
                    elif o == 2: word = {'m': 'два', 'f': 'две', 'n': 'два'}[gender]
                    parts.append(word)
        return " ".join(parts)

    def transliterate(text):
        clusters = {'sh': 'ш', 'ch': 'ч', 'th': 'т', 'ph': 'ф', 'kh': 'х', 'zh': 'ж', 'oo': 'у', 'ee': 'и'}
        mapping = {'a': 'а', 'b': 'б', 'c': 'к', 'd': 'д', 'e': 'е', 'f': 'ф', 'g': 'г', 'h': 'х', 'i': 'и', 'j': 'й', 'k': 'к', 'l': 'л', 'm': 'м', 'n': 'н', 'o': 'о', 'p': 'п', 'q': 'к', 'r': 'р', 's': 'с', 't': 'т', 'u': 'у', 'v': 'в', 'w': 'в', 'x': 'кс', 'y': 'и', 'z': 'з'}
        text = text.lower()
        res = []
        i = 0
        while i < len(text):
            found = False
            if i + 1 < len(text) and text[i:i+2] in clusters:
                cyr = clusters[text[i:i+2]]
                for c in cyr: res.append(c + "_trans")
                i += 2
                found = True
            if not found:
                char = text[i]
                if char in mapping:
                    cyr = mapping[char]
                    for c in cyr: res.append(c + "_trans")
                else: res.append(char)
                i += 1
        return " ".join(res)

    def transform_token(before, class_name):
        b = str(before)
        if b == '<PAD>': return 'sil'
        
        # Dictionary Overrides
        if (class_name, b) in class_tag_lookup: return class_tag_lookup[(class_name, b)]
        if b in global_lookup: return global_lookup[b]
        
        if class_name == 'PUNCT': return b
        
        if class_name in ['CARDINAL', 'DIGIT']:
            if b.isdigit(): return num_to_ru(int(b))
            
        if class_name == 'DECIMAL':
            if '.' in b or ',' in b:
                parts = b.replace(',', '.').split('.')
                if len(parts) == 2:
                    int_val, frac_str = int(parts[0]), parts[1]
                    int_word = num_to_ru(int_val, gender='f')
                    int_unit = get_plural_adj(int_val, ['целая', 'целых'])
                    frac_val = int(frac_str)
                    frac_word = num_to_ru(frac_val, gender='f')
                    scale = {1: ['десятая', 'десятых'], 2: ['сотая', 'сотых'], 3: ['тысячная', 'тысячных']}.get(len(frac_str), ['десятитысячная', 'десятитысячных'])
                    frac_unit = get_plural_adj(frac_val, scale)
                    return f"{int_word} {int_unit} {frac_word} {frac_unit}"

        if class_name == 'TIME':
            match = re.match(r'(\d{1,2})[:.](\d{2})', b)
            if match:
                h, m = int(match.group(1)), int(match.group(2))
                h_str = f"{num_to_ru(h, gender='m')} {get_plural(h, ['час', 'часа', 'часов'])}"
                m_str = f"{num_to_ru(m, gender='f')} {get_plural(m, ['минута', 'минуты', 'минут'])}"
                return f"{h_str} {m_str}"

        if class_name == 'VERBATIM':
            symbols = {'%': 'процент', '&': 'и', '$': 'доллар', '№': 'номер', '+': 'плюс', '-': 'минус', '*': 'умножить'}
            return symbols.get(b, b)

        if class_name in ['LETTERS', 'ELECTRONIC', 'PLAIN']:
            if re.search(r'[a-zA-Z]', b): return transliterate(b)

        return b

    print("Applying linguistic transformation...")
    val_final = [transform_token(b, id_to_class[c]) for b, c in zip(val_strings, val_class_preds)]
    test_final = [transform_token(b, id_to_class[c]) for b, c in zip(test_strings, test_class_preds)]

    val_preds_series = pd.Series(val_final, index=X_val.index.to_numpy())
    test_preds_series = pd.Series(test_final)

    print(f"Validation Class Accuracy: {np.mean(val_class_preds == y_val.to_numpy()):.4f}")
    return val_preds_series, test_preds_series

# ===== Model Registry =====
PREDICTION_ENGINES: Dict[str, ModelFn] = {
    "phonetic_xgb_engine": train_phonetic_xgb_engine,
}