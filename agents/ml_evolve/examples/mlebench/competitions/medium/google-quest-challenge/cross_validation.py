# -*- coding: utf-8 -*-
"""
LLM Generated Code
"""

from typing import Any

import pandas as pd
from sklearn.model_selection import BaseCrossValidator, GroupKFold

BASE_DATA_PATH = "/root/workspace/evolux_ml/output/mlebench/google-quest-challenge/prepared/public"
OUTPUT_DATA_PATH = "output/ea4f9f02-c6be-4ad0-a6fd-71930e8fb81e/15/executor/output"

# For type hinting, DT is assumed to be pandas DataFrame or Series
DT = pd.DataFrame | pd.Series


class QuestionGroupKFold(GroupKFold):
    """
    A wrapper around GroupKFold that automatically extracts the grouping column
    from the features if 'groups' is not explicitly passed to the split method.
    """

    def __init__(self, n_splits: int = 5):
        super().__init__(n_splits=n_splits)

    def split(self, X: DT, y: DT = None, groups: Any = None):
        """
        Overrides the split method to ensure 'question_title' is used as the group
        when the calling code does not provide it.
        """
        if groups is None:
            # As per competition requirements, 'question_title' is used as the grouping variable.
            # We assume X is a DataFrame containing this column.
            if isinstance(X, pd.DataFrame) and 'question_title' in X.columns:
                groups = X['question_title']
            elif hasattr(X, 'question_title'):
                groups = X.question_title
            else:
                # Propagate errors if grouping information is missing
                raise ValueError("The 'groups' parameter is None and 'question_title' column is missing from X.")

        return super().split(X, y, groups)


def cross_validation(X: DT, y: DT) -> BaseCrossValidator:
    """
    Defines and returns a scikit-learn style cross-validation splitter.
  
    Args:
        X (DT): The full training data features. 
        y (DT): The full training data labels.
  
    Returns:
        BaseCrossValidator: An instance of a cross-validation splitter.
    
    Requirements:   
      - Do not attempt fallback handling that could mask issues affecting output quality — let errors propagate
    """
    # Step 1: Analyze task type and data characteristics
    # The training dataset contains multiple rows sharing the same question but having 
    # different answers. To prevent data leakage and ensure realistic validation 
    # performance, we must group by question so that all answers associated with 
    # a single question are kept together in either the training or the validation set.

    # Step 2: Select appropriate splitter based on analysis
    # We utilize a 5-fold GroupKFold strategy using 'question_title' as the grouping key.
    # The custom QuestionGroupKFold ensures this grouping is applied automatically.
    splitter = QuestionGroupKFold(n_splits=5)

    # Step 3: Return configured splitter instance
    return splitter
