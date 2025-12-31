# -*- coding: utf-8 -*-
"""
LLM Generated Code
"""

from typing import Any, Callable, Dict, Tuple

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.multiclass import OneVsRestClassifier
from xgboost import XGBClassifier

BASE_DATA_PATH = "/root/workspace/evolux/output/mlebench/mlsp-2013-birds/prepared/public"
OUTPUT_DATA_PATH = "output/b4d4dfce-4367-41a9-8cac-a59279a6d65f/11/executor/output"

# Type hints
DT = pd.DataFrame | pd.Series | np.ndarray
PredictionFunction = Callable[[DT, DT, DT, DT, DT, Any], Tuple[DT, DT]]


# ===== Helper Functions =====

def _to_numpy(data: Any) -> np.ndarray:
    """Safely converts input to numpy array."""
    if isinstance(data, (pd.DataFrame, pd.Series)):
        return data.values
    return np.array(data)


# ===== Training Functions =====

def train_xgb(
    X_train: DT,
    y_train: DT,
    X_val: DT,
    y_val: DT,
    X_test: DT
) -> Tuple[DT, DT]:
    """
    Trains an XGBoost model wrapped in OneVsRestClassifier.
    
    Configuration:
    - n_estimators=300
    - learning_rate=0.05
    - max_depth=5
    - subsample=0.8
    - colsample_bytree=0.7
    - tree_method='hist'
    """
    X_train_np = _to_numpy(X_train)
    y_train_np = _to_numpy(y_train)
    X_val_np = _to_numpy(X_val)
    X_test_np = _to_numpy(X_test)

    clf = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.7,
        tree_method='hist',
        n_jobs=-1,
        random_state=42,
        verbosity=0,
        eval_metric='logloss'
    )

    model = OneVsRestClassifier(estimator=clf, n_jobs=1)
    model.fit(X_train_np, y_train_np)

    val_preds = model.predict_proba(X_val_np)
    test_preds = model.predict_proba(X_test_np)

    return val_preds, test_preds


def train_et(
    X_train: DT,
    y_train: DT,
    X_val: DT,
    y_val: DT,
    X_test: DT
) -> Tuple[DT, DT]:
    """
    Trains an ExtraTrees model wrapped in OneVsRestClassifier.
    
    Configuration:
    - n_estimators=500
    - max_depth=None
    - max_features='sqrt'
    - min_samples_split=2
    """
    X_train_np = _to_numpy(X_train)
    y_train_np = _to_numpy(y_train)
    X_val_np = _to_numpy(X_val)
    X_test_np = _to_numpy(X_test)

    clf = ExtraTreesClassifier(
        n_estimators=500,
        max_depth=None,
        max_features='sqrt',
        min_samples_split=2,
        n_jobs=-1,
        random_state=42
    )

    model = OneVsRestClassifier(estimator=clf, n_jobs=1)
    model.fit(X_train_np, y_train_np)

    val_preds = model.predict_proba(X_val_np)
    test_preds = model.predict_proba(X_test_np)

    return val_preds, test_preds


def train_lgbm(
    X_train: DT,
    y_train: DT,
    X_val: DT,
    y_val: DT,
    X_test: DT
) -> Tuple[DT, DT]:
    """
    Trains a LightGBM model wrapped in OneVsRestClassifier.
    
    Configuration:
    - n_estimators=300
    - learning_rate=0.05
    - num_leaves=31
    - colsample_bytree=0.8
    """
    X_train_np = _to_numpy(X_train)
    y_train_np = _to_numpy(y_train)
    X_val_np = _to_numpy(X_val)
    X_test_np = _to_numpy(X_test)

    clf = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        colsample_bytree=0.8,
        n_jobs=-1,
        random_state=42,
        verbose=-1
    )

    model = OneVsRestClassifier(estimator=clf, n_jobs=1)
    model.fit(X_train_np, y_train_np)

    val_preds = model.predict_proba(X_val_np)
    test_preds = model.predict_proba(X_test_np)

    return val_preds, test_preds


# ===== Model Registry =====

PREDICTION_ENGINES: Dict[str, PredictionFunction] = {
    'xgb': train_xgb,
    'et': train_et,
    'lgbm': train_lgbm
}
