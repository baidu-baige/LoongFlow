#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file provide base memory interface
"""

from __future__ import annotations

import math
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Dict, Optional, Tuple

import numpy as np


def clean_nan_values(obj: Any) -> Any:
    """
    Recursively clean NaN values from a data structure, replacing them with
    None. This ensures JSON serialization works correctly.
    """
    if isinstance(obj, dict):
        return {key: clean_nan_values(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [clean_nan_values(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(clean_nan_values(item) for item in obj)
    elif isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    elif isinstance(obj, np.floating) and (np.isnan(obj) or np.isinf(obj)):
        return None
    elif hasattr(obj, "dtype") and np.issubdtype(obj.dtype, np.floating):
        # Handle numpy arrays and scalars
        if np.isscalar(obj):
            if np.isnan(obj) or np.isinf(obj):
                return None
            else:
                return float(obj)
        else:
            # For numpy arrays, convert to list and clean recursively
            return clean_nan_values(obj.tolist())
    else:
        return obj


@dataclass
class Solution:
    """Represent a solution in the memory."""

    # Solution identification
    solution: str = ""
    solution_id: str = ""

    # Evolution information
    generate_plan: str = ""
    parent_id: Optional[str] = ""
    island_id: Optional[int] = 0
    iteration: Optional[int] = 0
    timestamp: float = field(default_factory=time.time)
    generation: int = 0
    sample_cnt: int = 0
    sample_weight: float = 0.0

    # Performance metrics
    score: Optional[float] = 0.0
    evaluation: Optional[str] = ""
    summary: str = ""

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def copy(self):
        """Return a copy of the solution."""
        return self.__class__(**{f.name: getattr(self, f.name) for f in fields(self)})

    def update(self, *args: Any, **kwargs: Any) -> None:
        """Update the current solution."""
        for arg in args:
            if isinstance(arg, dict):
                for key, value in arg.items():
                    setattr(self, key, value)
            elif isinstance(arg, Solution):
                for field in fields(self):
                    if getattr(arg, field.name) is not None:
                        setattr(self, field.name, getattr(arg, field.name))
        for key, value in kwargs.items():
            setattr(self, key, value)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        data = asdict(self)
        return clean_nan_values(data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Solution":
        """Create from dictionary representation"""
        # Get the valid field names for the Program dataclass
        valid_fields = {f.name for f in fields(cls)}

        # Filter the data to only include valid fields
        if isinstance(data, dict):
            filtered_data = {k: v for k, v in data.items() if k in valid_fields}
        else:
            # Handle case where data is not a dictionary
            filtered_data = {}

        # Ensure required fields have values
        required_fields = {"score", "timestamp", "solution"}
        for field in required_fields:
            if field not in filtered_data:
                if field == "score":
                    filtered_data[field] = 0.0
                elif field == "timestamp":
                    filtered_data[field] = time.time()
                elif field == "solution":
                    filtered_data[field] = ""

        # Ensure island_id is properly converted to int
        if "island_id" in filtered_data and filtered_data["island_id"] is not None:
            try:
                filtered_data["island_id"] = int(filtered_data["island_id"])
            except (TypeError, ValueError):
                filtered_data["island_id"] = None

        return cls(**filtered_data)


class EvolveMemory(ABC):
    """Abstract base class for EvolveMemory implementations.

    Evolutionary memory stores content(eg. Solution) for a specific memory, allowing
    agents to maintain context without requiring explicit manual memory management.

    This ABC is intended for internal use and as a base class for concrete implementations.
    Third-party libraries should implement the EvolveMemory protocol instead.
    """

    def __init__(self):
        self.memory_id: str = uuid.uuid4().hex[:8]

        self.diversity_reference_size: int = 20
        self.diversity_cache_size: int = 1000
        self.feature_scaling_method: str = "minmax"

    @abstractmethod
    async def add_solution(self, *args: Any, **kwargs: Any) -> str:
        """Add the solution for this session."""
        ...

    @abstractmethod
    def get_solutions(self, *args: Any, **kwargs: Any) -> list[Solution]:
        """Get solutions from the memory by solution ids."""
        ...

    @abstractmethod
    def list_solutions(self, *args: Any, **kwargs: Any) -> list[Solution]:
        """List solutions from the memory by created time."""
        ...

    @abstractmethod
    def get_best_solutions(self, *args: Any, **kwargs: Any) -> list[Solution]:
        """Get the best solutions from the memory by some criteria, like island_id or topK."""
        ...

    @abstractmethod
    def sample(self, *args: Any, **kwargs: Any) -> Solution:
        """Sample a solution from the memory."""
        ...

    @abstractmethod
    async def save_checkpoint(self, *args: Any, **kwargs: Any) -> None:
        """Save the current state of the memory to persistent storage."""
        ...

    @abstractmethod
    def load_checkpoint(self, *args: Any, **kwargs: Any) -> None:
        """Load the memory state from the last saved checkpoint."""
        ...

    @abstractmethod
    def memory_status(self, *args: Any, **kwargs: Any) -> dict:
        """Return the status of the memory"""
        ...

    @abstractmethod
    async def update_solution(self, *args: Any, **kwargs: Any) -> None:
        """Update a solution's properties in the memory."""
        ...

    def _calculate_feature_coords(
        self,
        solution: Solution,
        solutions: Dict[str, Solution],
        feature_stats: Dict[str, Dict[str, float | float | list[float]]],
        feature_bins_per_dim: Dict[str, int],
        feature_bins: int,
        feature_dimensions: list[str],
        diversity_cache: Dict[str, Dict[str, float]],
        diversity_reference_set: list[str],
    ) -> Tuple[Dict[str, int], list[str]]:
        """
        Calculate MAP-Elites feature coordinates for a solution with caching.

        Args:
            solution: The solution to calculate features for
            solutions: Dictionary of all solutions in memory
            feature_stats: Statistics for feature scaling
            feature_bins_per_dim: Number of bins per feature dimension
            diversity_cache: Cache for diversity calculations
            diversity_reference_set: Reference set for diversity calculation

        Returns:
            Tuple containing:
            - List of feature coordinates (bin indices)
            - Updated diversity reference set

        Raises:
            ValueError: If feature dimension is invalid or missing required data
        """
        if not solution.solution:
            raise ValueError("Solution content cannot be empty")

        coords = {}
        updated_ref_set = diversity_reference_set

        for dim in feature_dimensions:
            if dim == "complexity":
                value = float(len(solution.solution))
                bin_idx = self._calculate_feature_bin(
                    dim, value, feature_stats, feature_bins_per_dim, feature_bins
                )
                coords[dim] = bin_idx
            elif dim == "diversity":
                if len(solutions) < 2:
                    bin_idx = 0  # Default value for small populations
                else:
                    diversity, updated_ref_set = self._get_cached_diversity(
                        solution, solutions, diversity_cache, updated_ref_set
                    )
                    bin_idx = self._calculate_feature_bin(
                        dim,
                        diversity,
                        feature_stats,
                        feature_bins_per_dim,
                        feature_bins,
                    )
                coords[dim] = bin_idx
            elif dim == "score":
                avg_score = solution.score
                # Update stats and scale
                self._update_feature_stats(feature_stats, "score", avg_score)
                scaled_value = self._scale_feature_value(
                    feature_stats, "score", avg_score
                )
                num_bins = feature_bins_per_dim.get("score", feature_bins)
                bin_idx = int(scaled_value * num_bins)
                bin_idx = max(0, min(num_bins - 1, bin_idx))
                coords[dim] = bin_idx
            else:
                raise ValueError(f"Unsupported feature dimension: {dim}")

        return coords, updated_ref_set

    def _calculate_feature_bin(
        self,
        feature_name: str,
        value: float,
        feature_stats: Dict[str, Dict[str, float | float | list[float]]],
        feature_bins_per_dim: Dict[str, int],
        feature_bins: int,
    ) -> int:
        """
        Calculate the bin index for a given complexity value using feature scaling.

        Args:
            feature_name: Feature dimension name
            value: Feature value to bin
            feature_stats: Statistics for feature scaling
            feature_bins_per_dim: Number of bins per dimension

        Returns:
            Bin index in range [0, self.feature_bins - 1]
        """
        # Update feature statistics
        self._update_feature_stats(feature_stats, feature_name, value)

        # Scale the value using configured method
        scaled_value = self._scale_feature_value(feature_stats, feature_name, value)

        # Get number of bins for this dimension
        num_bins = feature_bins_per_dim.get(feature_name, feature_bins)

        # Convert to bin index
        bin_idx = int(scaled_value * num_bins)

        # Ensure bin index is within valid range
        bin_idx = max(0, min(num_bins - 1, bin_idx))

        return bin_idx

    def _update_feature_stats(
        self,
        feature_stats: Dict[str, Dict[str, float | float | list[float]]],
        feature_name: str,
        value: float,
    ) -> None:
        """
        Update statistics for a feature dimension

        Args:
            feature_stats: Feature statistics dictionary
            feature_name: Name of the feature dimension
            value: New value to incorporate into stats
        """

        if feature_name not in feature_stats:
            feature_stats[feature_name] = {
                "min": value,
                "max": value,
                "values": [],  # Keep recent values for percentile calculation if needed
            }

        stats = feature_stats[feature_name]
        stats["min"] = min(float(stats["min"]), value)
        stats["max"] = max(float(stats["max"]), value)

        # Keep recent values for more sophisticated scaling methods
        stats["values"].append(value)
        if len(stats["values"]) > 1000:  # Limit memory usage
            stats["values"] = stats["values"][-1000:]

    def _serialize_feature_stats(
        self, feature_stats: Dict[str, Dict[str, float | float | list[float]]]
    ) -> Dict[str, Any]:
        """
        Serialize feature_stats for JSON storage

        Returns:
            Dictionary that can be JSON-serialized
        """
        serialized = {}
        for feature_name, stats in feature_stats.items():
            # Convert to JSON-serializable format
            serialized_stats = {}
            for key, value in stats.items():
                if key == "values":
                    # Limit size to prevent excessive memory usage
                    # Keep only the most recent 100 values for percentile calculations
                    if isinstance(value, list) and len(value) > 100:
                        serialized_stats[key] = value[-100:]
                    else:
                        serialized_stats[key] = value
                else:
                    # Convert numpy types to Python native types
                    if hasattr(value, "item"):  # numpy scalar
                        serialized_stats[key] = value.item()
                    else:
                        serialized_stats[key] = value
            serialized[feature_name] = serialized_stats
        return serialized

    def _deserialize_feature_stats(
        self, stats_dict: Dict[str, Any]
    ) -> Dict[str, Dict[str, float | list[float]]]:
        """
        Deserialize feature_stats from loaded JSON

        Args:
            stats_dict: Dictionary loaded from JSON

        Returns:
            Properly formatted feature_stats dictionary
        """
        if not stats_dict:
            return {}

        deserialized = {}
        for feature_name, stats in stats_dict.items():
            if isinstance(stats, dict):
                # Ensure proper structure and types
                deserialized_stats = {
                    "min": float(stats.get("min", 0.0)),
                    "max": float(stats.get("max", 1.0)),
                    "values": list(stats.get("values", [])),
                }
                deserialized[feature_name] = deserialized_stats

        return deserialized

    def _scale_feature_value(
        self,
        feature_stats: Dict[str, Dict[str, float | float | list[float]]],
        feature_name: str,
        value: float,
    ) -> float:
        """
        Scale a feature value according to the configured scaling method

        Args:
            feature_stats: Feature statistics dictionary
            feature_name: Name of the feature dimension
            value: Raw feature value

        Returns:
            Scaled value in range [0, 1]
        """
        if feature_name not in feature_stats:
            return min(1.0, max(0.0, value))

        stats = feature_stats[feature_name]

        if self.feature_scaling_method == "minmax":
            # Min-max normalization to [0, 1]
            min_val = stats["min"]
            max_val = stats["max"]

            if max_val == min_val:
                return 0.5  # All values are the same

            scaled = (value - min_val) / (max_val - min_val)
            return min(1.0, max(0.0, scaled))  # Ensure in [0, 1]

        elif self.feature_scaling_method == "percentile":
            # Use percentile ranking
            values = stats["values"]
            if not values:
                return 0.5

            # Count how many values are less than or equal to this value
            count = sum(1 for v in values if v <= value)
            percentile = count / len(values)
            return percentile

        else:
            if feature_name not in feature_stats:
                return min(1.0, max(0.0, value))

            stats = feature_stats[feature_name]
            min_val = stats["min"]
            max_val = stats["max"]

            if max_val == min_val:
                return 0.5

            scaled = (value - min_val) / (max_val - min_val)
            return min(1.0, max(0.0, scaled))

    def _get_cached_diversity(
        self,
        solution: Solution,
        solutions: Dict[str, Solution],
        diversity_cache: Dict[str, Dict[str, float]],
        diversity_reference_set: list[str],
    ) -> Tuple[float, list[str]]:
        """
        Get diversity score for a solution using cache and reference set

        Args:
            solution: The solution to calculate diversity for
            solutions: Dictionary of all solutions
            diversity_cache: Cache for diversity scores
            diversity_reference_set: Current reference set

        Returns:
            Tuple of (diversity_score, updated_reference_set)
        """
        code_hash = hash(solution.solution)

        # Check cache first
        if code_hash in diversity_cache:
            return diversity_cache[code_hash]["value"], diversity_reference_set

        # Update reference set if needed
        if (
            not diversity_reference_set
            or len(diversity_reference_set) < self.diversity_reference_size
        ):
            new_diversity_reference_set = self._update_diversity_reference_set(
                solutions
            )
        else:
            new_diversity_reference_set = diversity_reference_set

        # Compute diversity against reference set
        diversity_scores = []
        for ref_solution in new_diversity_reference_set:
            if ref_solution != solution.solution:  # Don't compare with itself
                diversity = self._fast_code_diversity(solution.solution, ref_solution)
                diversity_scores.append(diversity)

        diversity = (
            sum(diversity_scores) / max(1, len(diversity_scores))
            if diversity_scores
            else 0.0
        )

        # Cache the result with LRU eviction
        self._cache_diversity_value(code_hash, diversity, diversity_cache)

        return diversity, new_diversity_reference_set

    def _update_diversity_reference_set(
        self, solutions: Dict[str, Solution]
    ) -> list[str]:
        """
        Update and return a diverse reference set of solutions.

        Uses an optimized greedy algorithm to select the most diverse subset of solutions.
        Time complexity: O(n*k) where n is number of solutions and k is reference size.

        Args:
            solutions: Dictionary of solution_id to Solution objects

        Returns:
            List of solution strings representing the reference set

        Raises:
            ValueError: If solutions dictionary is empty
        """
        if not solutions:
            raise ValueError("Cannot update reference set from empty solutions")

        # Early return if we can take all solutions
        all_solutions = list(solutions.values())
        if len(all_solutions) <= self.diversity_reference_size:
            return [s.solution for s in all_solutions]

        # Optimized selection process with memoization
        selected = []
        remaining = all_solutions.copy()

        # 1. Pre-compute all pairwise diversities
        diversity_matrix = {}
        for i in range(len(remaining)):
            for j in range(i + 1, len(remaining)):
                key = (remaining[i].solution_id, remaining[j].solution_id)
                diversity_matrix[key] = self._fast_code_diversity(
                    remaining[i].solution, remaining[j].solution
                )

        # 2. Initialize with two most diverse solutions
        if len(remaining) >= 2:
            max_pair = max(diversity_matrix.items(), key=lambda x: x[1])
            sol1_id, sol2_id = max_pair[0]
            selected.extend(
                [
                    next(s for s in remaining if s.solution_id == sol1_id),
                    next(s for s in remaining if s.solution_id == sol2_id),
                ]
            )
            remaining = [
                s for s in remaining if s.solution_id not in (sol1_id, sol2_id)
            ]

        # 3. Greedy selection with memoized diversities
        while len(selected) < self.diversity_reference_size and remaining:
            best_solution = max(
                remaining,
                key=lambda cand: min(
                    diversity_matrix.get(
                        tuple(sorted((cand.solution_id, sel.solution_id))),
                        self._fast_code_diversity(cand.solution, sel.solution),
                    )
                    for sel in selected
                ),
            )
            selected.append(best_solution)
            remaining.remove(best_solution)

        return [s.solution for s in selected]

    def _fast_code_diversity(self, code1: str, code2: str) -> float:
        """
        Efficiently calculate diversity score between two code solutions.

        Uses a combination of:
        1. Length difference (weight: 0.1)
        2. Line count difference (weight: 10)
        3. Character set difference (weight: 0.5)

        Args:
            code1: First code solution as string
            code2: Second code solution as string

        Returns:
            Diversity score between 0.0 (identical) and higher values (more diverse)

        Raises:
            ValueError: If either input is not a string
        """
        if not isinstance(code1, str) or not isinstance(code2, str):
            raise ValueError("Both inputs must be strings")

        if code1 == code2:
            return 0.0

        # Pre-compute frequently used values
        len1, len2 = len(code1), len(code2)
        lines1 = code1.count("\n")
        lines2 = code2.count("\n")

        # Calculate metrics with optimized weights
        length_diff = abs(len1 - len2) * 0.1
        line_diff = abs(lines1 - lines2) * 10

        # Optimized character set difference calculation
        if len1 < len2:
            smaller, larger = code1, code2
        else:
            smaller, larger = code2, code1

        unique_chars = set()
        for c in larger:
            if c not in smaller:
                unique_chars.add(c)

        char_diff = len(unique_chars) * 0.5

        # Combine weighted metrics
        return length_diff + line_diff + char_diff

    def _cache_diversity_value(
        self,
        code_hash: int,
        diversity: float,
        diversity_cache: Dict[str, Dict[str, float]],
    ) -> None:
        """Cache a diversity value with LRU eviction"""
        # Check if cache is full
        if len(diversity_cache) >= self.diversity_cache_size:
            # Remove oldest entry
            oldest_hash = min(diversity_cache.items(), key=lambda x: x[1]["timestamp"])[
                0
            ]
            del diversity_cache[oldest_hash]

        # Add new entry
        diversity_cache[code_hash] = {"value": diversity, "timestamp": time.time()}

    def _feature_coords_to_key(self, coords: Dict[str, int]) -> str:
        """
        Convert feature coordinates to a string key

        Args:
            coords: Feature coordinates

        Returns:
            String key
        """
        values = [v for v in coords.values()]
        return "-".join(str(c) for c in values)

    def _is_better(self, solution1: Solution, solution2: Solution) -> bool:
        """
        Determine if solution1 is better than solution2

        Args:
            solution1: First solution
            solution2: Second solution

        Returns:
            True if solution1 is better than solution2
        """
        # If no metrics, use newest
        if not solution1.score and not solution2.score:
            return solution1.timestamp > solution2.timestamp

        # If only one has metrics, it's better
        if solution1.score and not solution2.score:
            return True
        if not solution1.score and solution2.score:
            return False

        return solution1.score > solution2.score
