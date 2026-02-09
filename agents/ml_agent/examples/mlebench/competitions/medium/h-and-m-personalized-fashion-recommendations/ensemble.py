import cudf
import cupy as cp
import pandas as pd
import numpy as np
import os
import gc
from typing import Dict, Any

# Task-adaptive type definitions using GPU-accelerated cuDF
y = cudf.Series
Predictions = cudf.DataFrame

BASE_DATA_PATH = "/mnt/pfs/loongflow/devmachine/h20-01/evolux/output/mlebench/h-and-m-personalized-fashion-recommendations/prepared/public"
OUTPUT_DATA_PATH = "output/083259f0-3b2a-44c4-af6c-8557ef06ad6a/8/executor/output"

def ensemble(
    all_val_preds: Dict[str, Predictions],
    all_test_preds: Dict[str, Predictions],
    y_val: y,
) -> Predictions:
    """
    Combines predictions from multiple models and ensures exactly 12 predictions per customer 
    using age-stratified popularity fallbacks according to Technical Specification.
    """
    print("Starting ensemble stage...")

    # 1. Aggregate Model Scores for Test Set
    print("Aggregating model predictions...")
    model_names = list(all_test_preds.keys())
    if not model_names:
        raise ValueError("No predictions found in all_test_preds.")

    # Technical Specification: Direct Ranking of Candidate DataFrames
    # Use a priority weight map for ensemble if multiple models are present.
    weight_map = {
        'lgbm_lambdarank_fusion': 1.0
    }
    
    ensemble_df = None
    for name in model_names:
        print(f"Processing model outputs: {name}")
        model_df = all_test_preds[name][['customer_id', 'article_id', 'score']].copy()
        weight = weight_map.get(name, 1.0 / len(model_names))
        model_df['score'] = (model_df['score'] * weight).astype('float32')
        
        if ensemble_df is None:
            ensemble_df = model_df
        else:
            # Atomic outer join to merge scores for common candidates
            ensemble_df = ensemble_df.merge(
                model_df, on=['customer_id', 'article_id'], how='outer', suffixes=('', '_new')
            ).fillna(0)
            ensemble_df['score'] = ensemble_df['score'] + ensemble_df['score_new']
            ensemble_df.drop(columns=['score_new'], inplace=True)
            del model_df
            gc.collect()

    # 2. Rank and Selection (Top 12)
    print("Ranking candidates and extracting top 12 per customer...")
    ensemble_df = ensemble_df.sort_values(['customer_id', 'score'], ascending=[True, False])
    ensemble_df['rank'] = ensemble_df.groupby('customer_id').cumcount()
    top_12 = ensemble_df[ensemble_df['rank'] < 12][['customer_id', 'article_id']].copy()
    del ensemble_df
    gc.collect()

    # 3. Age-Stratified Fallback Popularity
    print("Calculating age-stratified popularity fallback (2020-09-08 to 2020-09-14)...")
    
    # Load customer metadata for age bins
    customers = cudf.read_csv(os.path.join(BASE_DATA_PATH, "customers.csv"), usecols=['customer_id', 'age'])
    age_median = customers['age'].median()
    customers['age'] = customers['age'].fillna(age_median)
    
    # Define demographic bins: Young (<25), Adult (25-40), Mature (40-60), Senior (60+)
    bins = [0, 25, 40, 60, 110]
    labels = [0, 1, 2, 3] 
    customers['age_bin'] = cudf.cut(customers['age'], bins=bins, labels=labels, right=False).astype('int8')
    
    # Identify popular items from the final week of training data
    trans_cols = ['t_dat', 'customer_id', 'article_id']
    transactions = cudf.read_csv(os.path.join(BASE_DATA_PATH, "transactions_train.csv"), usecols=trans_cols)
    transactions['t_dat'] = cudf.to_datetime(transactions['t_dat'])
    recent_trans = transactions[transactions['t_dat'] >= '2020-09-08'].copy()
    del transactions
    gc.collect()
    
    # Compute popularity per age bin
    recent_trans = recent_trans.merge(customers[['customer_id', 'age_bin']], on='customer_id', how='left')
    recent_trans['age_bin'] = recent_trans['age_bin'].fillna(1).astype('int8') # Default to bin 1 (Adult)
    
    age_pop = recent_trans.groupby(['age_bin', 'article_id']).size().reset_index(name='count')
    age_pop = age_pop.sort_values(['age_bin', 'count'], ascending=[True, False])
    age_pop_top12 = age_pop.groupby('age_bin').head(12).reset_index(drop=True)
    
    # Global top 12 for absolute cold-start safety
    global_top12 = recent_trans['article_id'].value_counts().head(12).index.to_arrow().to_pylist()
    del recent_trans
    gc.collect()

    # Create mapping for stratified fallback
    age_pop_pd = age_pop_top12.to_pandas()
    age_pop_dict = age_pop_pd.groupby('age_bin')['article_id'].apply(list).to_dict()

    # 4. Final Formatting and Padding
    print("Preparing final submission with gap-filling logic...")
    sample_sub = cudf.read_csv(os.path.join(BASE_DATA_PATH, "sample_submission.csv"), usecols=['customer_id'])
    
    # Convert top_12 to pandas for grouping and formatting (Technical Specification requirement)
    top_12_pd = top_12.to_pandas()
    grouped_preds = top_12_pd.groupby('customer_id')['article_id'].apply(list).reset_index()
    
    # Merge with sample submission to include all required customers
    full_preds = sample_sub.to_pandas().merge(grouped_preds, on='customer_id', how='left')
    full_preds = full_preds.merge(customers.to_pandas()[['customer_id', 'age_bin']], on='customer_id', how='left')
    full_preds['age_bin'] = full_preds['age_bin'].fillna(1).astype('int8')
    
    del top_12, top_12_pd, grouped_preds
    gc.collect()

    def fill_and_format(row):
        """Ensures exactly 12 unique predictions per customer with stratified fallback."""
        preds = row['article_id']
        if not isinstance(preds, list):
            preds = []
        
        # Priority 1: Model predictions (already in preds)
        # Priority 2: Age-stratified popularity
        age_bin = row['age_bin']
        fallback = age_pop_dict.get(age_bin, global_top12)
        
        for item in fallback:
            if len(preds) >= 12:
                break
            if item not in preds:
                preds.append(item)
        
        # Priority 3: Global popularity (if still under 12)
        if len(preds) < 12:
            for item in global_top12:
                if len(preds) >= 12:
                    break
                if item not in preds:
                    preds.append(item)
        
        # Return space-separated string of integer IDs
        # Casting to int and then str ensures no scientific notation or decimals
        return " ".join(map(str, preds[:12]))

    print("Generating space-separated article_id strings...")
    full_preds['prediction'] = full_preds.apply(fill_and_format, axis=1)
    
    # Final conversion back to cuDF for output
    submission_df = cudf.from_pandas(full_preds[['customer_id', 'prediction']])

    # 5. Persistence
    os.makedirs(OUTPUT_DATA_PATH, exist_ok=True)
    out_path = os.path.join(OUTPUT_DATA_PATH, "submission.csv")
    submission_df.to_csv(out_path, index=False)
    
    print(f"Ensemble module complete. Submission file saved to {out_path}.")
    print(f"Total customers processed: {len(submission_df)}")
    
    return submission_df