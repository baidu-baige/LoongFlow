import lightgbm as lgb
import cudf
import numpy as np
import pandas as pd
import gc
from typing import Tuple, Any, Dict, Callable

# Define concrete types for this task
X = cudf.DataFrame
y = cudf.Series
Predictions = cudf.DataFrame

# Model Function Type
ModelFn = Callable[
    [X, y, X, y, X],
    Tuple[Predictions, Predictions]
]

# ===== Training Functions =====

def train_lgbm_lambdarank_fusion(
    X_train: X,
    y_train: y,
    X_val: X,
    y_val: y,
    X_test: X
) -> Tuple[Predictions, Predictions]:
    """
    Trains a high-capacity LightGBM LambdaRank model with GPU acceleration.
    Rank the candidate pool using LambdaRank while integrating the external fusion signal.
    
    Handles potential empty inputs from upstream preprocessing (e.g., in validation_mode).
    """
    print("Starting LightGBM LambdaRank training (Fusion Strategy)...")

    # Step 1: Feature Selection
    # Exclude identifiers and potential target leakage columns
    id_cols = ['customer_id', 'article_id', 'label', '_y', '_label']
    feature_cols = [col for col in X_train.columns if col not in id_cols]
    
    # Check for fusion signal column (Technical Specification: fusion_score_ad7ccf68)
    fusion_col = 'fusion_score_ad7ccf68'
    if fusion_col in feature_cols:
        print(f"Fusion signal '{fusion_col}' detected and included as feature.")
    
    # Safety Check: If data is empty or no features exist, return zero-score predictions
    if len(X_train) == 0 or len(X_val) == 0 or not feature_cols:
        print("Warning: Training or Validation data is empty, or no features were found. Returning zero scores.")
        val_preds = X_val[['customer_id', 'article_id']].copy()
        val_preds['score'] = 0.0
        test_preds = X_test[['customer_id', 'article_id']].copy()
        test_preds['score'] = 0.0
        return val_preds, test_preds

    print(f"Total features used: {len(feature_cols)}")

    # Step 2: Prepare Training Data
    # LambdaRank requires data to be grouped by query (customer_id).
    def get_lgb_dataset(df: X, labels: y, reference=None):
        # Create a copy to avoid modifying original and for sorting
        temp = df.copy()
        temp['_label'] = labels
        # Sort by customer_id to ensure query groups are contiguous
        temp = temp.sort_values('customer_id').reset_index(drop=True)
        
        # Calculate group sizes
        q_sizes = temp.groupby('customer_id', sort=False).size().to_pandas().values
        
        # Convert to Pandas for LightGBM compatibility
        # H20 hardware (440GB RAM) allows efficient conversion even for large data
        ds = lgb.Dataset(
            temp[feature_cols].to_pandas(),
            label=temp['_label'].to_pandas(),
            group=q_sizes,
            reference=reference,
            free_raw_data=False
        )
        del temp
        gc.collect()
        return ds

    print("Constructing LightGBM Datasets...")
    dtrain = get_lgb_dataset(X_train, y_train)
    dval = get_lgb_dataset(X_val, y_val, reference=dtrain)

    # Step 3: Configure Hyperparameters (Technical Specification)
    params = {
        'objective': 'lambdarank',
        'metric': 'map',
        'eval_at': [12],
        'num_leaves': 63,            # Specified
        'learning_rate': 0.03,       # Specified
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'importance_type': 'gain',
        'device': 'cuda',            # GPU Acceleration
        'gpu_use_dp': False,
        'verbose': -1,
        'seed': 42,
        'feature_pre_filter': False
    }

    # Step 4: Execute Training
    print("Training LambdaRank on GPU...")
    model = lgb.train(
        params,
        dtrain,
        num_boost_round=1500,
        valid_sets=[dtrain, dval],
        valid_names=['train', 'valid'],
        callbacks=[
            lgb.early_stopping(stopping_rounds=100),
            lgb.log_evaluation(period=100)
        ]
    )

    # Step 5: Generate Predictions
    print("Generating ranking scores...")
    
    # Predict on original order to ensure alignment with input DataFrames
    # We use chunks if necessary, but with 440GB RAM, direct prediction is usually safe
    val_scores = model.predict(X_val[feature_cols].to_pandas())
    test_scores = model.predict(X_test[feature_cols].to_pandas())

    # Sanitize outputs (No NaNs/Infs)
    val_scores = np.nan_to_num(val_scores, nan=0.0, posinf=0.0, neginf=0.0)
    test_scores = np.nan_to_num(test_scores, nan=0.0, posinf=0.0, neginf=0.0)

    # Package into Predictions DataFrames
    val_preds = X_val[['customer_id', 'article_id']].copy()
    val_preds['score'] = val_scores.astype('float32')
    
    test_preds = X_test[['customer_id', 'article_id']].copy()
    test_preds['score'] = test_scores.astype('float32')

    # Final Cleanup
    del dtrain, dval, model
    gc.collect()

    print(f"LambdaRank Fusion complete. Validation Rows: {len(val_preds)}, Test Rows: {len(test_preds)}")
    
    return val_preds, test_preds


# ===== Model Registry =====
# Register the LambdaRank fusion engine
PREDICTION_ENGINES: Dict[str, ModelFn] = {
    "lgbm_lambdarank_fusion": train_lgbm_lambdarank_fusion,
}