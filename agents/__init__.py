"""LoongFlow Agents - Pre-built agent implementations.

This package provides ready-to-use agent implementations for specific domains:
- Mathematical optimization and discovery
- Machine learning competitions
- General evolutionary algorithms
"""

from . import general_evolve
from . import ml_evolve

__all__ = ["general_evolve", "ml_evolve"]