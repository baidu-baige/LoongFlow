from typing import List, Dict, Any
import numpy as np
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

BASE_DATA_PATH = "/mnt/pfs/loongflow/devmachine/h20-10/evolux/output/mlebench/mlsp-2013-birds/prepared/public"
OUTPUT_DATA_PATH = "output/9f7a14b2-9e2e-4beb-a8af-238199431c62/57/executor/output"

# Task-adaptive type definitions aligned with upstream load_data component
X = List[Dict[str, Any]]  # List of dictionaries containing multi-modal features per recording
y = np.ndarray            # (N, 19) float32 array of multi-labels

def get_splitter(X: X, y: y) -> Any:
    """
    Defines and returns a data splitting strategy for model validation.

    The bird species classification task involves 19 labels with extreme label sparsity 
    (some species occur only a few times). MultilabelStratifiedKFold is utilized to 
    preserve the distribution of multiple labels simultaneously across folds, which 
    is critical for valid performance estimation on rare classes and avoiding 
    folds that lack specific labels entirely.

    Args:
        X (X): The full training features.
        y (y): The full training targets (Multi-label binary matrix).

    Returns:
        MultilabelStratifiedKFold: A splitter object that implements:
            - split(X, y=None, groups=None) -> Iterator[(train_idx, val_idx)]
            - get_n_splits() -> int
    """
    # MultilabelStratifiedKFold is required because standard StratifiedKFold 
    # cannot handle multi-label (matrix) targets.
    # Configuration:
    # - n_splits=5: 5-fold cross-validation
    # - shuffle=True: Ensures random assignment to folds
    # - random_state=42: Ensures reproducibility across runs
    splitter = MultilabelStratifiedKFold(
        n_splits=5, 
        shuffle=True, 
        random_state=42
    )
    
    return splitter