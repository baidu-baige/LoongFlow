# -*- coding: utf-8 -*-
"""
LLM Generated Code
"""

import re
from typing import Any, Callable, Dict, Tuple

import numpy as np
import pandas as pd

BASE_DATA_PATH = "/root/workspace/evolux/output/mlebench/text-normalization-challenge-russian-language/prepared/public"
OUTPUT_DATA_PATH = "output/dce1a922-fb4b-4006-9d03-6f53b7ea0718/1/executor/output"

# Type hints
DT = pd.DataFrame | pd.Series | np.ndarray
PredictionFunction = Callable[[DT, DT, DT, DT, DT, Any], Tuple[DT, DT]]


class RussianTextNormalizer:
    """
    A lookup-based text normalizer for Russian language.
    Uses training data to build comprehensive lookup tables and
    applies rule-based transformations for unseen tokens.
    """

    def __init__(self):
        self.before_to_after = {}  # Direct lookup: before -> most common after
        self.before_to_after_by_context = {}  # Context-aware lookup
        self.class_patterns = {}  # Patterns for each class

        # Russian number words
        self.units = ['', 'один', 'два', 'три', 'четыре', 'пять', 'шесть', 'семь', 'восемь', 'девять']
        self.units_fem = ['', 'одна', 'две', 'три', 'четыре', 'пять', 'шесть', 'семь', 'восемь', 'девять']
        self.teens = ['десять', 'одиннадцать', 'двенадцать', 'тринадцать', 'четырнадцать',
                      'пятнадцать', 'шестнадцать', 'семнадцать', 'восемнадцать', 'девятнадцать']
        self.tens = ['', '', 'двадцать', 'тридцать', 'сорок', 'пятьдесят',
                     'шестьдесят', 'семьдесят', 'восемьдесят', 'девяносто']
        self.hundreds = ['', 'сто', 'двести', 'триста', 'четыреста', 'пятьсот',
                         'шестьсот', 'семьсот', 'восемьсот', 'девятьсот']

        # Ordinal forms (nominative masculine)
        self.ordinal_units = ['', 'первый', 'второй', 'третий', 'четвёртый', 'пятый',
                              'шестой', 'седьмой', 'восьмой', 'девятый']
        self.ordinal_teens = ['десятый', 'одиннадцатый', 'двенадцатый', 'тринадцатый',
                              'четырнадцатый', 'пятнадцатый', 'шестнадцатый',
                              'семнадцатый', 'восемнадцатый', 'девятнадцатый']
        self.ordinal_tens = ['', '', 'двадцатый', 'тридцатый', 'сороковой', 'пятидесятый',
                             'шестидесятый', 'семидесятый', 'восьмидесятый', 'девяностый']
        self.ordinal_hundreds = ['', 'сотый', 'двухсотый', 'трёхсотый', 'четырёхсотый',
                                 'пятисотый', 'шестисотый', 'семисотый', 'восьмисотый', 'девятисотый']

        # Patterns for classification
        self.punct_pattern = re.compile(r'^[.,;:!?\-—–()[\]{}«»"\'\s…]+$')
        self.numeric_pattern = re.compile(r'^\d+$')
        self.date_pattern = re.compile(
            r'\d{1,2}\s*(января|февраля|марта|апреля|мая|июня|июля|августа|сентября|октября|ноября|декабря)',
            re.IGNORECASE)
        self.year_pattern = re.compile(r'^\d{4}\s*(год|года|году)?$', re.IGNORECASE)
        self.time_pattern = re.compile(r'^\d{1,2}:\d{2}(:\d{2})?$')
        self.roman_pattern = re.compile(r'^[IVXLCDM]+$', re.IGNORECASE)
        self.latin_pattern = re.compile(r'[a-zA-Z]')
        self.cyrillic_pattern = re.compile(r'[а-яА-ЯёЁ]')

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        """Build lookup tables from training data."""
        print("Building lookup tables...")

        # Combine for easier processing
        train_data = X_train.copy()
        train_data['after'] = y_train.values

        # Build primary lookup: before -> most common after
        before_after_counts = train_data.groupby(['before', 'after']).size().reset_index(name='count')
        idx = before_after_counts.groupby('before')['count'].idxmax()
        most_common = before_after_counts.loc[idx]

        for _, row in most_common.iterrows():
            self.before_to_after[row['before']] = row['after']

        print(f"  Built primary lookup with {len(self.before_to_after)} entries")

        # Build context-aware lookup for ambiguous tokens
        # Key: (prev_token, before, next_token) -> after
        train_data = train_data.sort_values(['sentence_id', 'token_id']).reset_index(drop=True)
        train_data['prev_before'] = train_data.groupby('sentence_id')['before'].shift(1).fillna('<START>')
        train_data['next_before'] = train_data.groupby('sentence_id')['before'].shift(-1).fillna('<END>')

        # Find tokens with multiple possible transformations
        ambiguous_tokens = before_after_counts.groupby('before').size()
        ambiguous_tokens = ambiguous_tokens[ambiguous_tokens > 1].index.tolist()

        # Build context lookup for ambiguous tokens
        ambiguous_data = train_data[train_data['before'].isin(ambiguous_tokens)]
        context_counts = ambiguous_data.groupby(['prev_before', 'before', 'next_before', 'after']).size().reset_index(
            name='count')

        for before_token in ambiguous_tokens[:10000]:  # Limit to top 10k ambiguous tokens
            token_contexts = context_counts[context_counts['before'] == before_token]
            if len(token_contexts) > 0:
                for _, row in token_contexts.iterrows():
                    key = (row['prev_before'], row['before'], row['next_before'])
                    if key not in self.before_to_after_by_context:
                        self.before_to_after_by_context[key] = (row['after'], row['count'])
                    elif row['count'] > self.before_to_after_by_context[key][1]:
                        self.before_to_after_by_context[key] = (row['after'], row['count'])

        print(f"  Built context lookup with {len(self.before_to_after_by_context)} entries")

        return self

    def _classify_token(self, token: str) -> str:
        """Classify a token into a category."""
        if not token or pd.isna(token):
            return 'PLAIN'

        token = str(token)

        if self.punct_pattern.match(token):
            return 'PUNCT'
        if self.time_pattern.match(token):
            return 'TIME'
        if self.date_pattern.search(token):
            return 'DATE'
        if self.year_pattern.match(token):
            return 'DATE'
        if self.roman_pattern.match(token):
            return 'ORDINAL'
        if self.numeric_pattern.match(token):
            return 'CARDINAL'
        if self.latin_pattern.search(token) and not self.cyrillic_pattern.search(token):
            return 'ELECTRONIC'

        return 'PLAIN'

    def _number_to_words(self, n: int, feminine: bool = False) -> str:
        """Convert a number to Russian words."""
        if n == 0:
            return 'ноль'

        if n < 0:
            return 'минус ' + self._number_to_words(-n, feminine)

        result = []

        # Billions
        if n >= 1000000000:
            billions = n // 1000000000
            n %= 1000000000
            result.append(self._number_to_words(billions, False))
            if billions % 10 == 1 and billions % 100 != 11:
                result.append('миллиард')
            elif 2 <= billions % 10 <= 4 and not (12 <= billions % 100 <= 14):
                result.append('миллиарда')
            else:
                result.append('миллиардов')

        # Millions
        if n >= 1000000:
            millions = n // 1000000
            n %= 1000000
            result.append(self._number_to_words(millions, False))
            if millions % 10 == 1 and millions % 100 != 11:
                result.append('миллион')
            elif 2 <= millions % 10 <= 4 and not (12 <= millions % 100 <= 14):
                result.append('миллиона')
            else:
                result.append('миллионов')

        # Thousands (feminine in Russian)
        if n >= 1000:
            thousands = n // 1000
            n %= 1000
            result.append(self._number_to_words(thousands, True))
            if thousands % 10 == 1 and thousands % 100 != 11:
                result.append('тысяча')
            elif 2 <= thousands % 10 <= 4 and not (12 <= thousands % 100 <= 14):
                result.append('тысячи')
            else:
                result.append('тысяч')

        # Hundreds
        if n >= 100:
            result.append(self.hundreds[n // 100])
            n %= 100

        # Tens and units
        if n >= 20:
            result.append(self.tens[n // 10])
            n %= 10
        elif n >= 10:
            result.append(self.teens[n - 10])
            n = 0

        if n > 0:
            if feminine:
                result.append(self.units_fem[n])
            else:
                result.append(self.units[n])

        return ' '.join(filter(None, result))

    def _transform_cardinal(self, token: str) -> str:
        """Transform a cardinal number."""
        # Extract digits
        digits = re.sub(r'[^\d]', '', token)
        if not digits:
            return token

        try:
            num = int(digits)
            return self._number_to_words(num)
        except:
            return token

    def _spell_out_digits(self, token: str) -> str:
        """Spell out each digit individually."""
        digit_words = ['ноль', 'один', 'два', 'три', 'четыре',
                       'пять', 'шесть', 'семь', 'восемь', 'девять']
        result = []
        for char in token:
            if char.isdigit():
                result.append(digit_words[int(char)])
            else:
                result.append(char)
        return ' '.join(result)

    def _transliterate(self, token: str) -> str:
        """Transliterate Latin characters to Russian _trans format."""
        # Simple transliteration mapping
        trans_map = {
            'a': 'а', 'b': 'б', 'c': 'к', 'd': 'д', 'e': 'е', 'f': 'ф',
            'g': 'г', 'h': 'х', 'i': 'и', 'j': 'дж', 'k': 'к', 'l': 'л',
            'm': 'м', 'n': 'н', 'o': 'о', 'p': 'п', 'q': 'к', 'r': 'р',
            's': 'с', 't': 'т', 'u': 'у', 'v': 'в', 'w': 'в', 'x': 'кс',
            'y': 'и', 'z': 'з'
        }

        result = []
        for char in token.lower():
            if char in trans_map:
                result.append(trans_map[char] + '_trans')
            elif char == '.':
                result.append('точка')
            elif char == '-':
                result.append('дефис')
            elif char.isdigit():
                result.append(self._number_to_words(int(char)))
            else:
                result.append(char)

        return ' '.join(result)

    def predict_single(self, token: str, prev_token: str = '<START>', next_token: str = '<END>') -> str:
        """Predict the normalized form of a single token."""
        if not token or pd.isna(token):
            return ''

        token = str(token)

        # Try context-aware lookup first
        context_key = (prev_token, token, next_token)
        if context_key in self.before_to_after_by_context:
            return self.before_to_after_by_context[context_key][0]

        # Try direct lookup
        if token in self.before_to_after:
            return self.before_to_after[token]

        # Fallback: classify and apply rules
        token_class = self._classify_token(token)

        if token_class == 'PUNCT':
            return token

        if token_class == 'PLAIN':
            return token

        if token_class == 'CARDINAL':
            return self._transform_cardinal(token)

        if token_class == 'ELECTRONIC':
            return self._transliterate(token)

        # Default: return as-is
        return token

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict normalized forms for all tokens."""
        # Sort by sentence_id and token_id
        X_sorted = X.sort_values(['sentence_id', 'token_id']).reset_index(drop=True)

        # Add context columns
        X_sorted['prev_before'] = X_sorted.groupby('sentence_id')['before'].shift(1).fillna('<START>')
        X_sorted['next_before'] = X_sorted.groupby('sentence_id')['before'].shift(-1).fillna('<END>')

        # Predict
        predictions = []
        for _, row in X_sorted.iterrows():
            pred = self.predict_single(
                row['before'],
                row['prev_before'],
                row['next_before']
            )
            predictions.append(pred)

        # Restore original order
        X_sorted['prediction'] = predictions
        X_sorted = X_sorted.sort_index()

        return X_sorted['prediction'].values


def train_lookup_normalizer(
    X_train: DT,
    y_train: DT,
    X_val: DT,
    y_val: DT,
    X_test: DT,
    **hyper_params: Any
) -> Tuple[DT, DT]:
    """
    Trains a lookup-based text normalizer and returns predictions.
    
    This approach is optimal for text normalization because:
    1. Most transformations are deterministic
    2. Exact string match is required
    3. Large training data provides comprehensive coverage
    
    Args:
        X_train: Training features with 'before', 'sentence_id', 'token_id' columns
        y_train: Training labels (normalized text)
        X_val: Validation features
        y_val: Validation labels
        X_test: Test features
        **hyper_params: Additional parameters (unused)
    
    Returns:
        Tuple of (validation_predictions, test_predictions)
    """
    print("Training lookup-based text normalizer...")

    # Convert to DataFrames if needed
    if isinstance(X_train, np.ndarray):
        X_train = pd.DataFrame(X_train)
    if isinstance(X_val, np.ndarray):
        X_val = pd.DataFrame(X_val)
    if isinstance(X_test, np.ndarray):
        X_test = pd.DataFrame(X_test)
    if isinstance(y_train, np.ndarray):
        y_train = pd.Series(y_train)

    # Initialize and fit normalizer
    normalizer = RussianTextNormalizer()
    normalizer.fit(X_train, y_train)

    # Predict on validation set
    print("Predicting on validation set...")
    val_predictions = normalizer.predict(X_val)

    # Calculate validation accuracy
    if y_val is not None:
        y_val_arr = y_val.values if isinstance(y_val, pd.Series) else y_val
        accuracy = np.mean(val_predictions == y_val_arr)
        print(f"Validation accuracy: {accuracy:.4f}")

    # Predict on test set
    print("Predicting on test set...")
    test_predictions = normalizer.predict(X_test)

    # Ensure no NaN values
    val_predictions = np.array([p if p and not pd.isna(p) else '' for p in val_predictions])
    test_predictions = np.array([p if p and not pd.isna(p) else '' for p in test_predictions])

    # Verify output requirements
    assert len(val_predictions) == len(X_val), f"Val prediction length mismatch: {len(val_predictions)} vs {len(X_val)}"
    assert len(test_predictions) == len(
        X_test), f"Test prediction length mismatch: {len(test_predictions)} vs {len(X_test)}"

    print(f"Predictions complete: {len(val_predictions)} val, {len(test_predictions)} test")

    return val_predictions, test_predictions


# ===== Model Registry =====
PREDICTION_ENGINES: Dict[str, PredictionFunction] = {
    "lookup_normalizer": train_lookup_normalizer,
}
