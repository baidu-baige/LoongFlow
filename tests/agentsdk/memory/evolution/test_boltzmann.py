import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../../.."))

from src.agentsdk.memory.evolution.boltzmann import _boltzmann_selection_with_weights


class Solution:
    def __init__(self, score=None, sample_weight=None, solution=None):
        self.score = score
        self.sample_weight = sample_weight
        self.solution = solution

    def __repr__(self):
        return f"Solution(score={self.score}, weight={self.sample_weight})"


class TestBoltzmannSelection:
    @pytest.fixture
    def sample_solutions(self):
        return [
            Solution(score=1.0, sample_weight=1.0, solution="sol1"),
            Solution(score=2.0, sample_weight=1.0, solution="sol2"),
            Solution(score=3.0, sample_weight=1.0, solution="sol3"),
            Solution(score=1.0, sample_weight=2.0, solution="sol4"),
            Solution(score=2.0, sample_weight=2.0, solution="sol5"),
            Solution(score=3.0, sample_weight=2.0, solution="sol6"),
        ]

    def test_basic_selection(self, sample_solutions):
        """Test basic selection behavior"""
        selected = _boltzmann_selection_with_weights(
            solutions=sample_solutions,
            tournament_size=3,
            temperature=1.0,
            use_sampling_weight=False,
        )
        assert selected in sample_solutions

    def test_score_priority(self, sample_solutions):
        """Test the priority based on score"""
        selections = []
        for _ in range(1000):
            selected = _boltzmann_selection_with_weights(
                solutions=sample_solutions[:3],
                tournament_size=3,
                temperature=1.0,
                use_sampling_weight=False,
            )
            selections.append(selected.score)

        assert np.mean(selections) > 2.0

    def test_weight_priority(self, sample_solutions):
        """Test that higher weight is more likely to be chosen"""
        selections = []
        for _ in range(1000):
            selected = _boltzmann_selection_with_weights(
                solutions=[s for s in sample_solutions if s.score == 2.0],
                tournament_size=3,
                temperature=1.0,
                use_sampling_weight=True,
            )
            selections.append(selected.sample_weight)

        assert np.mean(selections) > 1.5

    def test_temperature_effect(self, sample_solutions):
        """Test the effect of the temperature parameter"""
        high_temp_selections = []
        low_temp_selections = []

        for _ in range(500):
            selected = _boltzmann_selection_with_weights(
                solutions=sample_solutions[:3],
                tournament_size=3,
                temperature=2.0,
                use_sampling_weight=False,
            )
            high_temp_selections.append(selected.score)

            selected = _boltzmann_selection_with_weights(
                solutions=sample_solutions[:3],
                tournament_size=3,
                temperature=0.5,
                use_sampling_weight=False,
            )
            low_temp_selections.append(selected.score)

        assert np.mean(low_temp_selections) > np.mean(high_temp_selections)

    def test_edge_cases(self):
        """Test edge cases"""
        with pytest.raises(ValueError):
            _boltzmann_selection_with_weights([], 3, 1.0)

        with pytest.raises(ValueError):
            _boltzmann_selection_with_weights([Solution(score=1.0)], 3, -1.0)

        sol = Solution(score=1.0)
        assert _boltzmann_selection_with_weights([sol], 3, 1.0) == sol
