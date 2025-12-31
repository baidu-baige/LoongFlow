# -*- coding: utf-8 -*-
"""
LLM Generated Code
"""

import gc
import re
from typing import Dict, List

import inflect
import numpy as np
import pandas as pd

from load_data import load_data

# Type Hints
DT = pd.DataFrame | pd.Series | np.ndarray


def ensemble(
    all_oof_preds: Dict[str, DT],
    all_test_preds: Dict[str, List[DT]],
    y_true_full: DT
) -> DT:
    """
    Ensembles predictions from multiple models and converts predicted classes 
    into final normalized strings using a high-precision hierarchical engine.
    """
    # 1. Data Alignment and Loading
    # Determine mode based on sample size to load corresponding features
    is_val_mode = len(y_true_full) < 1000000
    X_train, _, X_test, _ = load_data(validation_mode=is_val_mode)

    # 2. Build Lookup Tables (Tier 1 and Tier 2)
    # y_true_full contains ['after', 'class']
    train_df = pd.DataFrame({
        'before': X_train['before'].values.astype(str),
        'after': y_true_full['after'].values.astype(str),
        'class': y_true_full['class'].values.astype(str)
    })

    # Tier 1: (class, before) -> after (most frequent mapping)
    t1_counts = train_df.groupby(['class', 'before', 'after']).size().reset_index(name='cnt')
    t1_map = \
        t1_counts.sort_values('cnt', ascending=False).drop_duplicates(['class', 'before']).set_index(
            ['class', 'before'])[
            'after'].to_dict()

    # Tier 2: Global Lookup Fallback (before -> after, most frequent mapping)
    t2_counts = train_df.groupby(['before', 'after']).size().reset_index(name='cnt')
    t2_map = t2_counts.sort_values('cnt', ascending=False).drop_duplicates('before').set_index('before')[
        'after'].to_dict()

    del train_df, t1_counts, t2_counts
    gc.collect()

    # 3. Aggregate Test Predictions (Majority Vote on Classes)
    all_preds_flat = []
    for model_name in all_test_preds:
        for f_pred in all_test_preds[model_name]:
            all_preds_flat.append(np.array(f_pred).astype(str))

    preds_df = pd.DataFrame({i: pred for i, pred in enumerate(all_preds_flat)})
    final_classes = preds_df.mode(axis=1)[0].values.astype(str)

    del preds_df, all_preds_flat
    gc.collect()

    # 4. Refined Rule Engine (Tier 3)
    inf_engine = inflect.engine()

    def clean_inf(text: str) -> str:
        """Mandatory cleaning for all inflect outputs."""
        if not text: return ""
        return text.replace(' and ', ' ').replace('-', ' ').replace(',', '').lower().strip()

    def rule_year(val_int: int) -> str:
        if val_int == 2000: return "two thousand"
        if 2000 < val_int < 2010:
            return "two thousand " + clean_inf(inf_engine.number_to_words(val_int % 100))
        high, low = divmod(val_int, 100)
        h_str = clean_inf(inf_engine.number_to_words(high))
        if low == 0: return h_str + " hundred"
        l_str = clean_inf(inf_engine.number_to_words(low))
        if low < 10: return h_str + " oh " + l_str
        return h_str + " " + l_str

    def rule_date(b: str) -> str | None:
        # YYYY
        if re.match(r'^\d{4}$', b):
            try:
                return rule_year(int(b))
            except:
                return None
        # Month DD
        m_md = re.match(r'^([A-Za-z]+)\s+(\d{1,2})(st|nd|rd|th)?$', b)
        if m_md:
            month, day = m_md.group(1), m_md.group(2)
            try:
                d_ord = clean_inf(inf_engine.number_to_words(inf_engine.ordinal(day)))
                return f"{month.lower()} {d_ord}"
            except:
                return None
        # DD/MM/YYYY or MM/DD/YYYY or YYYY-MM-DD
        m_iso = re.match(r'^(\d{4})-(\d{1,2})-(\d{1,2})$', b)
        m_dmy = re.match(r'^(\d{1,2})[./](\d{1,2})[./](\d{2,4})$', b)
        months = ["", "january", "february", "march", "april", "may", "june", "july", "august", "september", "october",
                  "november", "december"]
        if m_iso:
            y, m, d = m_iso.groups()
            try:
                return f"the {clean_inf(inf_engine.number_to_words(inf_engine.ordinal(d)))} of {months[int(m)]} {rule_year(int(y))}"
            except:
                pass
        if m_dmy:
            d, m, y = m_dmy.groups()
            try:
                y_val = int(y)
                if y_val < 100: y_val += 2000 if y_val < 50 else 1900
                return f"the {clean_inf(inf_engine.number_to_words(inf_engine.ordinal(d)))} of {months[int(m)]} {rule_year(y_val)}"
            except:
                pass
        return None

    def rule_money(b: str) -> str | None:
        curr_map = {'$': ('dollar', 'dollars'), '£': ('pound', 'pounds'), '€': ('euro', 'euros')}
        if not b or b[0] not in curr_map: return None
        sym = b[0]
        amt_str = b[1:].replace(',', '')
        if ' ' in amt_str:  # e.g. "$1 million"
            parts = amt_str.split(' ')
            try:
                num_w = clean_inf(inf_engine.number_to_words(parts[0]))
                return f"{num_w} {parts[1]} {curr_map[sym][1]}"
            except:
                return None
        try:
            if '.' in amt_str:
                major, minor = amt_str.split('.')
                major_v = int(major) if major else 0
                minor_v = int(minor) if minor else 0
                res = ""
                if major_v > 0:
                    res += clean_inf(inf_engine.number_to_words(major_v)) + " " + (
                        curr_map[sym][0] if major_v == 1 else curr_map[sym][1])
                if minor_v > 0:
                    if res: res += " "
                    res += clean_inf(inf_engine.number_to_words(minor_v)) + (" cent" if minor_v == 1 else " cents")
                return res.strip()
            else:
                val = int(amt_str)
                return clean_inf(inf_engine.number_to_words(val)) + " " + (
                    curr_map[sym][0] if val == 1 else curr_map[sym][1])
        except:
            return None

    def rule_measure(b: str) -> str | None:
        m = re.findall(r'(\d+(?:\.\d+)?)([a-zA-Z%]+)', b)
        if not m: return None
        num_str, unit = m[0]
        unit_map = {'%': 'percent', 'ft': 'feet', 'lb': 'pounds', 'oz': 'ounces', 'km': 'kilometers', 'kg': 'kilograms',
                    'mph': 'miles per hour'}
        try:
            num_w = clean_inf(inf_engine.number_to_words(num_str))
            u_word = unit_map.get(unit.lower(), unit)
            return f"{num_w} {u_word}"
        except:
            return None

    def rule_fraction(b: str) -> str | None:
        if b == "1/2": return "one half"
        m = re.match(r'^(\d+)/(\d+)$', b)
        if m:
            try:
                num_w = clean_inf(inf_engine.number_to_words(m.group(1)))
                den_w = clean_inf(inf_engine.number_to_words(m.group(2)))
                return f"{num_w} over {den_w}"
            except:
                pass
        return None

    def rule_verbatim(b: str) -> str:
        mapping = {'&': 'and', '#': 'number', '+': 'plus', '*': 'asterisk'}
        return mapping.get(b, b)

    def rule_digit_tele(b: str) -> str:
        mapping = {'0': 'o', '1': 'one', '2': 'two', '3': 'three', '4': 'four', '5': 'five', '6': 'six', '7': 'seven',
                   '8': 'eight', '9': 'nine'}
        res = []
        for char in b:
            if char in mapping:
                res.append(mapping[char])
            elif char in "-.() /":
                res.append("sil")
        return " ".join(res).strip()

    def rule_electronic(b: str) -> str:
        mapping = {'.': 'dot', ':': 'colon', '/': 'slash', '@': 'at', '_': 'underscore'}
        res = []
        for char in b.lower():
            res.append(mapping.get(char, char))
        return " ".join(res).strip()

    # 5. Apply Hierarchical Normalization
    test_before = X_test['before'].values.astype(str)
    if len(final_classes) < len(test_before):
        test_before = test_before[:len(final_classes)]

    results = []
    for b, c in zip(test_before, final_classes):
        # Tier 1: Class-Before Lookup
        res = t1_map.get((c, b))
        if res is not None:
            results.append(res)
            continue

        # Tier 2: Global Lookup Fallback
        res = t2_map.get(b)
        if res is not None:
            results.append(res)
            continue

        # Tier 3: Refined Rule Engine
        rule_res = None
        if c == 'DATE':
            rule_res = rule_date(b)
        elif c == 'MONEY':
            rule_res = rule_money(b)
        elif c == 'MEASURE':
            rule_res = rule_measure(b)
        elif c == 'FRACTION':
            rule_res = rule_fraction(b)
        elif c == 'VERBATIM':
            rule_res = rule_verbatim(b)
        elif c in ['TELEPHONE', 'DIGIT']:
            rule_res = rule_digit_tele(b)
        elif c == 'ELECTRONIC':
            rule_res = rule_electronic(b)
        elif c == 'ORDINAL':
            m = re.match(r'^(\d+)(st|nd|rd|th)$', b.lower())
            if m: rule_res = clean_inf(inf_engine.number_to_words(inf_engine.ordinal(m.group(1))))
        elif c == 'CARDINAL':
            try:
                rule_res = clean_inf(inf_engine.number_to_words(b.replace(',', '')))
            except:
                pass
        elif c == 'LETTERS':
            rule_res = " ".join(list(b.lower().replace(" ", "")))

        if rule_res:
            results.append(rule_res)
            continue

        # Tier 4: Identity Fallback
        results.append(b)

    return np.array(results)
