import numpy as np
import pytest


class Solution:
    def __init__(self, score=None, sample_weight=None, solution=None):
        self.score = score
        self.sample_weight = sample_weight
        self.solution = solution

    def __repr__(self):
        return f"Solution(score={self.score}, weight={self.sample_weight})"


def _boltzmann_selection_with_weights(
    solutions: list[Solution],
    tournament_size: int,
    temperature: float,
    use_sampling_weight: bool = True,
    sampling_weight_power: float = 1.0,
) -> Solution | None:
    """
    Copy from src/agentsdk/memory/evolution/boltzmann.py
    """
    if not solutions:
        raise ValueError("Cannot select from empty solutions list")
    if temperature <= 0:
        raise ValueError("Temperature must be positive")

    # Split into elite and non-elite groups
    sorted_solutions = sorted(
        solutions, key=lambda x: x.score or -float("inf"), reverse=True
    )
    elite_cutoff = int(len(sorted_solutions) * 0.3)
    elite = sorted_solutions[:elite_cutoff]
    non_elite = sorted_solutions[elite_cutoff:]

    # Sample candidates from both groups
    candidates = []
    if elite:
        candidates.extend(
            np.random.choice(elite, size=min(1, len(elite)), replace=False)
        )
    if non_elite:
        candidates.extend(
            np.random.choice(
                non_elite, size=min(tournament_size - 1, len(non_elite)), replace=False
            )
        )

    if not candidates:
        return None

    # Calculate Boltzmann probabilities
    scores = np.array([s.score or 0 for s in candidates])
    max_score = np.max(scores)

    # Boltzmann selection: exp((score - max_score) / temperature)
    # This ensures probabilities are normalized and numerically stable
    boltzmann_probs = np.exp((scores - max_score) / temperature)

    if use_sampling_weight:
        # Get sampling weights and apply power transformation
        sampling_weights = np.array([s.sample_weight or 1.0 for s in candidates])

        if sampling_weight_power != 1.0:
            sampling_weights = np.power(sampling_weights, sampling_weight_power)

        # Combine Boltzmann probabilities with sampling weights using proper normalization
        # This ensures the mathematical properties are preserved:
        # 1. Same weights: higher score → higher probability
        # 2. Same scores: higher weight → higher probability
        combined_probs = boltzmann_probs * sampling_weights

        # Validate and normalize probabilities
        combined_probs = np.nan_to_num(combined_probs, nan=0.0, posinf=0.0, neginf=0.0)
        combined_probs = np.clip(combined_probs, 0.0, None)
        probs_sum = np.sum(combined_probs)

        if probs_sum > 0 and not np.any(np.isnan(combined_probs)):
            # Normalize to create proper probability distribution
            normalized_probs = combined_probs / probs_sum
            return candidates[np.random.choice(len(candidates), p=normalized_probs)]
    else:
        # Standard Boltzmann selection without weights
        boltzmann_probs = np.nan_to_num(
            boltzmann_probs, nan=0.0, posinf=0.0, neginf=0.0
        )
        boltzmann_probs = np.clip(boltzmann_probs, 0.0, None)
        probs_sum = np.sum(boltzmann_probs)

        if probs_sum > 0 and not np.any(np.isnan(boltzmann_probs)):
            normalized_probs = boltzmann_probs / probs_sum
            return candidates[np.random.choice(len(candidates), p=normalized_probs)]

    # Fallback mechanisms for edge cases
    try:
        # Score-based fallback (softmax)
        scores = np.array([s.score or 0 for s in candidates])
        scores = np.clip(scores, -1e10, 1e10)
        scores = np.nan_to_num(scores, nan=0.0)
        score_probs = np.exp(scores - np.max(scores))
        return candidates[
            np.random.choice(len(candidates), p=score_probs / np.sum(score_probs))
        ]
    except:
        # Ultimate fallback - select highest score
        return max(candidates, key=lambda x: x.score or -float("inf"))


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
        """The high score is more likely to be chosen"""
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
        """The higher weight is more likely to be chosen"""
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
            # High temperature selection (exploration)
            selected = _boltzmann_selection_with_weights(
                solutions=sample_solutions[:3],
                tournament_size=3,
                temperature=2.0,
                use_sampling_weight=False,
            )
            high_temp_selections.append(selected.score)

            # Low temperature selection (exploitation)
            selected = _boltzmann_selection_with_weights(
                solutions=sample_solutions[:3],
                tournament_size=3,
                temperature=0.5,
                use_sampling_weight=False,
            )
            low_temp_selections.append(selected.score)

        # Test that low temperature favors higher scores
        assert np.mean(low_temp_selections) > np.mean(high_temp_selections)

    def test_edge_cases(self):
        """Test edge case handling"""
        with pytest.raises(ValueError):
            _boltzmann_selection_with_weights([], 3, 1.0)

        with pytest.raises(ValueError):
            _boltzmann_selection_with_weights([Solution(score=1.0)], 3, -1.0)

        sol = Solution(score=1.0)
        assert _boltzmann_selection_with_weights([sol], 3, 1.0) == sol
