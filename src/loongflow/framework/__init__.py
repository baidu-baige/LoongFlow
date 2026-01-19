"""Framework - Core agent architectures for LoongFlow.

This module provides the core implementations for different agent paradigms:
- EvolveAgent: Evolutionary algorithms for optimization
- ReActAgent: Standard reasoning loops
- Base classes for building custom agents
"""

from . import base
from . import evolve
from . import react

__all__ = ["base", "evolve", "react"]