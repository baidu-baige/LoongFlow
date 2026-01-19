"""ML Evolve - Machine Learning Agent Framework.

This module provides agents specialized for Machine Learning competitions
and AutoML tasks, including Kaggle competitions and MLE-Bench scenarios.
"""

from . import evaluator
from . import evocoder
from . import executor
from . import planner
from . import prompt
from . import summary
from . import utils

__all__ = ["evaluator", "evocoder", "executor", "planner", "prompt", "summary", "utils"]