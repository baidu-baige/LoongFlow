import os
import cudf
import pandas as pd
import numpy as np
import gc
from typing import Dict, Any

# Paths aligned with pipeline requirements
BASE_DATA_PATH = "/mnt/pfs/loongflow/devmachine/h20-12/evolux/output/mlebench/text-normalization-challenge-russian-language/prepared/public"
OUTPUT_DATA_PATH = "output/ecfe1a48-59fb-4170-a38b-6ffb4a298ec0/10/executor/output"

# Task-adaptive type definitions
y = cudf.Series           # Target vector (normalized strings)
Predictions = pd.Series   # Model predictions are normalized strings

def transliterate_engine(text: str) -> str:
    """
    Phonetically transliterates Latin strings into Russian characters with _trans suffixes.
    Handles clusters (sh, ch, etc.) and individual mapping.
    """
    if not isinstance(text, str) or text == '<NULL>':
        return text
    
    clusters = {
        'sh': 'ш', 'ch': 'ч', 'th': 'т', 'ph': 'ф', 
        'kh': 'х', 'zh': 'ж', 'oo': 'у', 'ee': 'и'
    }
    mapping = {
        'a': 'а', 'b': 'б', 'c': 'к', 'd': 'д', 'e': 'е', 'f': 'ф', 'g': 'г', 'h': 'х', 
        'i': 'и', 'j': 'й', 'k': 'к', 'l': 'л', 'm': 'м', 'n': 'н', 'o': 'о', 'p': 'п', 
        'q': 'к', 'r': 'р', 's': 'с', 't': 'т', 'u': 'у', 'v': 'в', 'w': 'в', 'x': 'кс', 
        'y': 'и', 'z': 'з'
    }
    
    text = text.lower()
    res = []
    i = 0
    while i < len(text):
        found = False
        # Check for 2-char clusters
        if i + 1 < len(text) and text[i:i+2] in clusters:
            cyr_chars = clusters[text[i:i+2]]
            for c in cyr_chars:
                res.append(c + "_trans")
            i += 2
            found = True
        
        if not found:
            char = text[i]
            if char in mapping:
                cyr_chars = mapping[char]
                for c in cyr_chars:
                    res.append(c + "_trans")
            else:
                # Keep punctuation or Cyrillic as is
                res.append(char)
            i += 1
            
    return " ".join(res)

def ensemble(
    all_val_preds: Dict[str, Predictions],
    all_test_preds: Dict[str, Predictions],
    y_val: y,
) -> Predictions:
    """
    Combines predictions using a Hierarchical Fallback strategy with a Maximum-Coverage Dict.
    
    Logic:
    1. Global Dictionary (>90% consistency lookup)
    2. Model + Rule Engine (XGBoost routed through transformation logic)
    3. Latin Fallback (Force transliteration for Latin tokens not in dict)
    4. Universal Fallback (Identity mapping)
    """
    print("Ensemble stage started...")

    if not all_test_preds:
        raise ValueError("No test predictions found in all_test_preds.")

    # 1. Evaluate individual model performance
    model_names = list(all_test_preds.keys())
    y_val_pd = y_val.to_pandas() if hasattr(y_val, 'to_pandas') else y_val
    
    for name in model_names:
        if name in all_val_preds:
            v_preds = all_val_preds[name]
            common_idx = v_preds.index.intersection(y_val_pd.index)
            if not common_idx.empty:
                acc = (v_preds.loc[common_idx] == y_val_pd.loc[common_idx]).mean()
                print(f"Model '{name}' OOF Accuracy: {acc:.6f}")

    # 2. Combine Base Model Predictions
    num_test_samples = len(all_test_preds[model_names[0]])
    if len(model_names) > 1:
        print(f"Ensembling {len(model_names)} models via majority voting...")
        test_preds_df = pd.DataFrame(all_test_preds)
        # Mode returns a DataFrame; take the first column
        ensembled_base = test_preds_df.mode(axis=1)[0]
    else:
        best_model = model_names[0]
        print(f"Using primary model output: {best_model}")
        ensembled_base = all_test_preds[best_model]

    # 3. Build Maximum-Coverage Global Dictionary (Consistency > 90%)
    train_csv = os.path.join(BASE_DATA_PATH, "prepared_optimized", "ru_train.csv")
    print(f"Building Global Dictionary from {train_csv}...")
    
    # Use GPU to process ~10M rows efficiently
    train_df = cudf.read_csv(
        train_csv, 
        usecols=['before', 'after'], 
        dtype={'before': 'string', 'after': 'string'}
    ).fillna('<NULL>')
    
    # Group by (before, after) to find the most frequent normalization per token
    counts = train_df.groupby(['before', 'after']).size().reset_index(name='pair_count')
    totals = counts.groupby('before')['pair_count'].transform('sum')
    counts['consistency'] = counts['pair_count'] / totals
    
    # Filter for high-confidence mappings (> 90%)
    reliable = counts[counts['consistency'] > 0.9]
    # Keep only the top mapping per token
    reliable = reliable.sort_values(by=['before', 'pair_count'], ascending=[True, False]).drop_duplicates('before')
    
    global_dict = reliable[['before', 'after']].to_pandas().set_index('before')['after'].to_dict()
    print(f"Global Dictionary size: {len(global_dict)} (90% consistency)")
    
    del train_df, counts, reliable
    gc.collect()

    # 4. Load Original Test Tokens for Hierarchical Fallback
    test_csv = os.path.join(BASE_DATA_PATH, "prepared_optimized", "ru_test_2.csv")
    print("Loading test tokens for priority routing...")
    # Handle possible validation subsetting
    test_before = cudf.read_csv(
        test_csv, 
        usecols=['before'], 
        dtype={'before': 'string'},
        nrows=num_test_samples
    ).fillna('<NULL>')['before'].to_pandas()

    # 5. Apply Hierarchy
    print("Applying Hierarchical Fallback Logic...")
    
    # Start with Global Dictionary lookups
    final_preds = test_before.map(global_dict)
    
    # Identify Latin-containing tokens
    latin_mask = test_before.str.contains(r'[a-zA-Z]', na=False)
    
    # Logic: 
    # - If in Dictionary: Use Dictionary
    # - Else if has Latin: Force Transliteration (Overrides generic model output if needed)
    # - Else: Use Model Prediction (which already includes rule-based logic)
    # - Else: Identity
    
    needs_filling = final_preds.isna()
    
    # Apply Latin Fallback override
    latin_fallback_mask = needs_filling & latin_mask
    if latin_fallback_mask.any():
        print(f"Applying Latin Fallback to {latin_fallback_mask.sum()} tokens...")
        final_preds.loc[latin_fallback_mask] = test_before.loc[latin_fallback_mask].apply(transliterate_engine)
    
    # Apply Model Predictions for the rest
    still_needs_filling = final_preds.isna()
    final_preds.loc[still_needs_filling] = ensembled_base.loc[still_needs_filling]
    
    # Final Universal Fallback: Identity
    final_preds = final_preds.fillna(test_before)

    # 6. Integrity and Cleanup
    if len(final_preds) != num_test_samples:
        raise ValueError(f"Ensemble size mismatch: {len(final_preds)} vs {num_test_samples}")
    
    if final_preds.isna().any():
        # This shouldn't happen due to identity fallback
        raise ValueError("NaN values detected in final ensemble output.")

    # Preserve the index of the model predictions (likely 0...N or test IDs)
    final_preds.index = ensembled_base.index
    
    print(f"Ensemble complete. Generated {len(final_preds)} predictions.")
    return final_preds