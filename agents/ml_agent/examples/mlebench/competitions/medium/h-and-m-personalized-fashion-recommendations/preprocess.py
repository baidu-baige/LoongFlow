import cudf
import cupy as cp
import pandas as pd
import os
import gc
from typing import Tuple

# Task-adaptive type definitions using GPU-accelerated cuDF
X = cudf.DataFrame
y = cudf.Series

def preprocess(
    X_train: X,
    y_train: y,
    X_val: X,
    y_val: y,
    X_test: X
) -> Tuple[X, y, X, y, X]:
    """
    Transforms raw data into model-ready candidate-based format using an Atomic Unified Pipeline.
    Strictly ensures row alignment, test coverage, and handles edge cases for empty data.
    """
    print("Starting Atomic Unified Preprocessing Pipeline...")

    # Technical Specification Parameters
    TRAIN_TARGET_START = '2020-09-01'
    VAL_TARGET_START = '2020-09-08'
    TEST_TARGET_START = '2020-09-15'

    # Global Metadata Extraction
    print("Extracting metadata...")
    articles_meta = cudf.concat([
        X_train[['article_id', 'product_type_name', 'index_name', 'graphical_appearance_name']],
        X_val[['article_id', 'product_type_name', 'index_name', 'graphical_appearance_name']]
    ]).drop_duplicates('article_id')
    
    customers_meta = cudf.concat([
        X_train[['customer_id', 'age', 'FN', 'Active']],
        X_test[['customer_id', 'age', 'FN', 'Active']]
    ]).drop_duplicates('customer_id')
    
    customers_meta['age_bin'] = cudf.cut(
        customers_meta['age'], 
        bins=[0, 20, 30, 40, 50, 60, 100], 
        labels=[0, 1, 2, 3, 4, 5]
    ).astype('int8')

    def generate_candidates(history_df: X, target_users: cudf.Series, now_date: str) -> X:
        """Multi-source recall (200-300 per user) with atomic safety."""
        if len(target_users) == 0:
            return cudf.DataFrame(columns=['customer_id', 'article_id'], dtype='int32').assign(customer_id=cudf.Series(dtype='str'))

        now = cudf.to_datetime(now_date)
        unique_targets = target_users.unique()
        target_df = unique_targets.to_frame(name='customer_id')
        all_cands = []
        
        # Determine global top items
        recent = history_df[history_df['t_dat'] >= (now - pd.to_timedelta(7, unit='D'))]
        if len(recent) == 0: recent = history_df.tail(1000)
        
        if len(recent) > 0:
            top_60 = recent['article_id'].value_counts().head(60).index.to_frame(name='article_id').reset_index(drop=True)
        else:
            # Absolute fallback
            top_60 = cudf.DataFrame({'article_id': [706016001, 706016002, 372860001]}, dtype='int32')
            
        # Source 1: Global Popularity (Primary coverage source)
        global_pop = target_df.assign(k=1).merge(top_60.assign(k=1), on='k').drop(columns='k')
        all_cands.append(global_pop)
        
        # Source 2: Age-Group Popularity
        if len(recent) > 0:
            recent_age = recent.merge(customers_meta[['customer_id', 'age_bin']], on='customer_id', how='left')
            recent_age['age_bin'] = recent_age['age_bin'].fillna(1).astype('int8')
            age_trends = recent_age.groupby(['age_bin', 'article_id']).size().reset_index(name='count')
            age_trends = age_trends.sort_values(['age_bin', 'count'], ascending=False).groupby('age_bin').head(40)
            
            target_age = target_df.merge(customers_meta[['customer_id', 'age_bin']], on='customer_id', how='left')
            target_age['age_bin'] = target_age['age_bin'].fillna(1).astype('int8')
            age_pop = target_age.merge(age_trends[['age_bin', 'article_id']], on='age_bin').drop(columns='age_bin')
            all_cands.append(age_pop)

        # Interaction-based sources
        hist_filtered = history_df.merge(target_df, on='customer_id', how='inner')
        if len(hist_filtered) > 0:
            # Source 3: Repurchase (4 months)
            repurchase = hist_filtered[hist_filtered['t_dat'] >= (now - pd.to_timedelta(120, unit='D'))]
            all_cands.append(repurchase[['customer_id', 'article_id']].drop_duplicates())
            
            # Source 4: Weighted CF (30 days)
            cf_window = history_df[history_df['t_dat'] >= (now - pd.to_timedelta(30, unit='D'))].copy()
            if len(cf_window) > 0:
                cf_window['w'] = 0.9 ** (now - cf_window['t_dat']).dt.days
                pairs = cf_window.merge(cf_window[['customer_id', 't_dat', 'article_id']], on=['customer_id', 't_dat'])
                pairs = pairs[pairs['article_id_x'] != pairs['article_id_y']]
                if len(pairs) > 0:
                    sims = pairs.groupby(['article_id_x', 'article_id_y'])['w'].sum().reset_index(name='sim')
                    top_sims = sims.sort_values(['article_id_x', 'sim'], ascending=False).groupby('article_id_x').head(10)
                    triggers = hist_filtered.sort_values(['customer_id', 't_dat']).groupby('customer_id').tail(5)[['customer_id', 'article_id']]
                    cf_candidates = triggers.merge(top_sims, left_on='article_id', right_on='article_id_x')[['customer_id', 'article_id_y']]
                    cf_candidates.columns = ['customer_id', 'article_id']
                    all_cands.append(cf_candidates)
            
            # Source 5: Last 20 interactions
            last_20 = hist_filtered.sort_values(['customer_id', 't_dat']).groupby('customer_id').tail(20)[['customer_id', 'article_id']]
            all_cands.append(last_20)

        # Atomic Merge and Deduplicate
        valid_dfs = [c for c in all_cands if len(c) > 0]
        if not valid_dfs:
            # This should never happen due to global_pop, but as a last resort:
            return global_pop 
        
        candidates = cudf.concat(valid_dfs).drop_duplicates(['customer_id', 'article_id'])
        return candidates.merge(target_df, on='customer_id', how='inner')

    def build_unified_df(history_df: X, target_users: cudf.Series, target_trans: X, now_date: str) -> Tuple[X, y]:
        """Atomic Unified Pipeline: Candidates -> Labels -> Features -> Return DF/y."""
        if len(target_users) == 0:
            cols = ['time_decay_count', 'style_affinity', 'index_affinity', 'item_velocity', 'user_avg_price_diff', 'age', 'FN', 'Active']
            empty_X = cudf.DataFrame(columns=['customer_id', 'article_id'] + cols)
            empty_y = cudf.Series(dtype='int8')
            return empty_X, empty_y

        now = cudf.to_datetime(now_date)
        cands = generate_candidates(history_df, target_users, now_date)
        
        # Atomic Labeling
        if target_trans is not None and len(target_trans) > 0:
            positives = target_trans[['customer_id', 'article_id']].drop_duplicates()
            positives['label'] = 1
            cands = cands.merge(positives, on=['customer_id', 'article_id'], how='left')
            cands['label'] = cands['label'].fillna(0).astype('int8')
        else:
            cands['label'] = 0

        # Feature Engineering
        cands = cands.merge(articles_meta, on='article_id', how='left')
        
        # time_decay_count
        h_decay = history_df[['customer_id', 'article_id', 't_dat']].copy()
        h_decay['days'] = (now - h_decay['t_dat']).dt.days
        h_decay['decay'] = 0.5 ** (h_decay['days'] / 7)
        f_decay = h_decay.groupby(['customer_id', 'article_id'])['decay'].sum().reset_index(name='time_decay_count')
        cands = cands.merge(f_decay, on=['customer_id', 'article_id'], how='left')
        
        # style/index affinity
        user_total = history_df.groupby('customer_id').size().reset_index(name='u_total')
        def get_aff(col, top_n, out_name):
            if history_df.empty: return cudf.DataFrame(columns=['customer_id', col, out_name])
            counts = history_df.groupby(['customer_id', col]).size().reset_index(name='c')
            counts = counts.merge(user_total, on='customer_id')
            counts['aff'] = counts['c'] / (counts['u_total'] + 1e-6)
            counts['r'] = counts.groupby('customer_id')['c'].rank(method='first', ascending=False)
            return counts[counts['r'] <= top_n][['customer_id', col, 'aff']].rename(columns={'aff': out_name})

        cands = cands.merge(get_aff('product_type_name', 3, 'style_affinity'), on=['customer_id', 'product_type_name'], how='left')
        cands = cands.merge(get_aff('index_name', 2, 'index_affinity'), on=['customer_id', 'index_name'], how='left')
        
        # item_velocity
        s7 = history_df[history_df['t_dat'] >= (now - pd.to_timedelta(7, unit='D'))].groupby('article_id').size().reset_index(name='c7')
        s3 = history_df[history_df['t_dat'] >= (now - pd.to_timedelta(3, unit='D'))].groupby('article_id').size().reset_index(name='c3')
        vel = s7.merge(s3, on='article_id', how='left').fillna(0)
        vel['item_velocity'] = vel['c3'] / (vel['c7'] + 1e-6)
        cands = cands.merge(vel[['article_id', 'item_velocity']], on='article_id', how='left')
        
        # user_avg_price_diff
        u_p = history_df.groupby('customer_id')['price'].mean().reset_index(name='u_p')
        i_p = history_df.groupby('article_id')['price'].mean().reset_index(name='i_p')
        cands = cands.merge(u_p, on='customer_id', how='left').merge(i_p, on='article_id', how='left')
        cands['user_avg_price_diff'] = cands['u_p'].fillna(0) - cands['i_p'].fillna(0)
        
        # Customer Metadata
        cands = cands.merge(customers_meta[['customer_id', 'age', 'FN', 'Active']], on='customer_id', how='left')
        cols = ['time_decay_count', 'style_affinity', 'index_affinity', 'item_velocity', 'user_avg_price_diff', 'age', 'FN', 'Active']
        cands[cols] = cands[cols].fillna(0).astype('float32')
        
        return cands[['customer_id', 'article_id'] + cols], cands['label']

    # --- Pipeline Execution ---
    print("Processing Training Set...")
    h_tr = X_train[X_train['t_dat'] < TRAIN_TARGET_START]
    t_tr = X_train[X_train['t_dat'] >= TRAIN_TARGET_START]
    if len(h_tr) == 0: h_tr = X_train.head(1000)
    X_tr_p, y_tr_p = build_unified_df(h_tr, t_tr['customer_id'].unique(), t_tr, TRAIN_TARGET_START)
    del h_tr, t_tr; gc.collect()

    print("Processing Validation Set...")
    X_val_p, y_val_p = build_unified_df(X_train, X_val['customer_id'].unique(), X_val, VAL_TARGET_START)
    gc.collect()

    print("Processing Test Set...")
    h_test_full = cudf.concat([X_train, X_val])
    target_test_ids = X_test['customer_id'].unique()
    X_test_p, _ = build_unified_df(h_test_full, target_test_ids, None, TEST_TARGET_START)
    y_test_p = cudf.Series(cp.zeros(len(X_test_p), dtype='int8'))
    del h_test_full; gc.collect()

    # Final Coverage Repair
    missing_ids = target_test_ids[~target_test_ids.isin(X_test_p['customer_id'])]
    if len(missing_ids) > 0:
        print(f"Repairing test coverage for {len(missing_ids)} users.")
        top_1 = X_train['article_id'].mode().iloc[0] if not X_train.empty else 706016001
        repair = missing_ids.to_frame(name='customer_id').assign(article_id=top_1)
        for col in ['time_decay_count', 'style_affinity', 'index_affinity', 'item_velocity', 'user_avg_price_diff', 'age', 'FN', 'Active']:
            repair[col] = 0.0
        X_test_p = cudf.concat([X_test_p, repair])
        y_test_p = cudf.Series(cp.zeros(len(X_test_p), dtype='int8'))

    print(f"Preprocessing complete. Train: {len(X_tr_p)}, Test: {len(X_test_p)}")
    return X_tr_p, y_tr_p, X_val_p, y_val_p, X_test_p