from typing import Tuple, Any, Dict, Callable
import numpy as np
import scipy.sparse as sp
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import SGDClassifier

BASE_DATA_PATH = "/mnt/pfs/loongflow/devmachine/h20-03/evolux/output/mlebench/facebook-recruiting-iii-keyword-extraction/prepared/public"
OUTPUT_DATA_PATH = "output/90689814-166b-4e4f-a971-355572d18239/1/executor/output"

# Task-adaptive concrete type definitions
X = sp.csr_matrix      # Sparse TF-IDF feature matrix
y = sp.csr_matrix      # Sparse binary indicator matrix for multi-label targets
Predictions = np.ndarray # Dense matrix of probability estimates (float32)

# Model Function Type
ModelFn = Callable[
    [X, y, X, y, X],
    Tuple[Predictions, Predictions]
]

# ===== Training Functions =====

def train_sgd_ovr(
    X_train: X,
    y_train: y,
    X_val: X,
    y_val: y,
    X_test: X
) -> Tuple[Predictions, Predictions]:
    """
    Trains a One-Vs-Rest meta-classifier using SGDClassifier as the base estimator.
    
    This strategy is designed for extreme multi-label classification (XML) at scale,
    utilizing sparse matrix operations and multi-core CPU parallelism.
    
    Args:
        X_train (X): Preprocessed sparse training features.
        y_train (y): Training targets as a sparse binary matrix.
        X_val (X): Preprocessed sparse validation features.
        y_val (y): Validation targets.
        X_test (X): Preprocessed sparse test features.

    Returns:
        Tuple[Predictions, Predictions]: Probability matrices for validation and test sets.
    """
    print(f"Initializing OneVsRest SGD model. Label space size: {y_train.shape[1]}")
    
    # Step 1: Build and configure model
    # SGDClassifier with log_loss provides probability estimates.
    # We use all 36 available cores via n_jobs=-1.
    # alpha=1e-5 is chosen for regularization balance in high-dimensional text space.
    base_clf = SGDClassifier(
        loss='log_loss', 
        penalty='l2', 
        alpha=1e-5, 
        random_state=42, 
        learning_rate='optimal'
    )
    
    model = OneVsRestClassifier(base_clf, n_jobs=-1)

    # Step 2: Enable GPU acceleration
    # Note: SGDClassifier and OneVsRestClassifier in scikit-learn are CPU-based.
    # We maximize utilization via n_jobs=-1 (36 cores).

    # Step 3: Train on (X_train, y_train)
    print("Starting training of 5000 binary classifiers in parallel...")
    model.fit(X_train, y_train)
    print("Training complete.")

    # Step 4: Predict on X_val and X_test
    print("Generating probability predictions for validation set...")
    # predict_proba returns a dense array of shape (n_samples, n_classes)
    val_preds = model.predict_proba(X_val).astype(np.float32)
    
    print("Generating probability predictions for test set...")
    test_preds = model.predict_proba(X_test).astype(np.float32)

    # Step 5: Final Verification
    if np.isnan(val_preds).any() or np.isinf(val_preds).any():
        raise ValueError("Validation predictions contain NaN or Infinity.")
    if np.isnan(test_preds).any() or np.isinf(test_preds).any():
        raise ValueError("Test predictions contain NaN or Infinity.")

    print(f"Predictions complete. Val shape: {val_preds.shape}, Test shape: {test_preds.shape}")
    
    return val_preds, test_preds


# ===== Model Registry =====
# Register ALL training functions here for the pipeline to use
PREDICTION_ENGINES: Dict[str, ModelFn] = {
    "sgd_ovr": train_sgd_ovr,
}