#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file provides Boltzmann selection strategy with adaptive temperature control.
"""

import numpy as np

from .base_memory import Solution


def _calculate_diversity(solutions: list[Solution], sample_size: int = 50) -> float:
    """
    Calculate the diversity of the population with optimized sampling.

    Args:
        solutions: list of solutions to calculate diversity for.
        sample_size: Number of solution pairs to sample for diversity estimation.

    Returns:
        Normalized diversity score between 0 (identical) and 1 (max diversity)

    Raises:
        ValueError: If solutions list is empty
    """
    if not solutions:
        raise ValueError("Cannot calculate diversity of empty solutions list")
    if len(solutions) <= 1:
        return 0.0

    # Sample solution pairs to reduce computation
    sample_indices = np.random.choice(
        len(solutions), size=min(sample_size, len(solutions)), replace=False
    )
    sampled_solutions = [solutions[i] for i in sample_indices]

    diversity_scores = []
    for i in range(len(sampled_solutions)):
        for j in range(i + 1, len(sampled_solutions)):
            s1 = sampled_solutions[i].solution or ""
            s2 = sampled_solutions[j].solution or ""

            # Use efficient diversity calculation
            len_diff = abs(len(s1) - len(s2)) / max(1, max(len(s1), len(s2)))
            line_diff = abs(s1.count("\n") - s2.count("\n")) / max(
                1, max(s1.count("\n"), s2.count("\n"))
            )
            char_diff = len(set(s1).symmetric_difference(set(s2))) / max(
                1, max(len(set(s1)), len(set(s2)))
            )

            # Combine metrics with weights
            combined_score = 0.4 * len_diff + 0.3 * line_diff + 0.3 * char_diff
            diversity_scores.append(combined_score)

    return np.mean(diversity_scores) if diversity_scores else 0.0


def _adaptive_temperature_by_diversity(
    current_temp: float,
    diversity: float,
    min_temp: float = 0.5,
    max_temp: float = 2.0,
    base_temp: float = 1.0,
) -> float:
    """
    Adaptively adjust temperature based on population diversity with smoother transitions.

    Args:
        current_temp: Current temperature value.
        diversity: Current diversity score (0-1).
        min_temp: Minimum allowed temperature.
        max_temp: Maximum allowed temperature.
        base_temp: Baseline temperature to scale from.

    Returns:
        Adjusted temperature value within [min_temp, max_temp] range.
    """
    # Sigmoid adjustment for smoother transitions
    adjustment_factor = 1 + (2 * diversity - 1)  # Maps 0-1 diversity to 0-2 adjustment

    # Apply adjustment with bounds
    new_temp = base_temp * adjustment_factor
    new_temp = max(min_temp, min(max_temp, new_temp))

    # Blend with current temperature for stability (20% weight to current)
    blended_temp = 0.8 * new_temp + 0.2 * current_temp

    return blended_temp


def select_parents_with_dynamic_temperature(
    solutions: list[Solution],
    elites: list[Solution],
    initial_temp: float,
    min_temp: float = 0.5,
    max_temp: float = 2.0,
    use_sampling_weight: bool = True,
    sampling_weight_power: float = 1.0,
    exploration_rate: float = 0.2,
) -> Solution | None:
    """
    Select parent solution with fully adaptive temperature control,
    optionally incorporating sampling weights with Boltzmann selection.

    Args:
        solutions: Population of candidate solutions.
        elites: Population of elite solutions.
        initial_temp: Starting temperature value.
        min_temp: Minimum allowed temperature.
        max_temp: Maximum allowed temperature.
        use_sampling_weight: Whether to incorporate sampling weights in selection.
        sampling_weight_power: Power to apply to sampling weights (1.0 = linear,
                              <1.0 = reduce influence, >1.0 = amplify influence).
        exploration_rate: Chance of selecting a random solution (0-1).

    Returns:
        Selected parent solution.

    Raises:
        ValueError: If solutions list is empty.
    """
    if not solutions:
        raise ValueError("Cannot select from empty solutions list")

    # Calculate current population diversity
    diversity = _calculate_diversity(solutions)

    # Adjust temperature based on diversity
    temperature = _adaptive_temperature_by_diversity(
        current_temp=initial_temp,
        diversity=diversity,
        min_temp=min_temp,
        max_temp=max_temp,
        base_temp=initial_temp,
    )

    return _boltzmann_selection_with_weights(
        solutions=solutions,
        elites=elites,
        temperature=temperature,
        use_sampling_weight=use_sampling_weight,
        sampling_weight_power=sampling_weight_power,
        exploration_rate=exploration_rate,
    )


def _boltzmann_selection_with_weights(
    solutions: list[Solution],
    elites: list[Solution],
    temperature: float,
    use_sampling_weight: bool = True,
    sampling_weight_power: float = 1.0,
    exploration_rate: float = 0.1,
) -> Solution | None:
    """
    Boltzmann selection with optional sampling weight integration and exploration.
    Features:
    - Same weights: higher score → higher probability
    - Same scores: higher weight → higher probability
    - exploration_rate: chance to select a random solution (0-1)

    Args:
        solutions: list of candidate solutions.
        elites: list of elite solutions.
        temperature: Current selection temperature.
        use_sampling_weight: Whether to use sampling weights in selection.
        sampling_weight_power: Power to apply to sampling weights.
        exploration_rate: Probability of selecting a random solution (0-1).

    Returns:
        Selected solution with balance of quality, diversity, and sampling weights.

    Raises:
        ValueError: If invalid inputs are provided.
    """
    if not solutions:
        raise ValueError("Cannot select from empty solutions list")
    if temperature <= 0:
        raise ValueError("Temperature must be positive")
    if not 0 <= exploration_rate <= 1:
        raise ValueError("exploration_rate must be between 0 and 1")

    # With exploration_rate probability, select a random solution
    if np.random.random() < exploration_rate:
        return np.random.choice(solutions)

    # Split into elite and non-elite groups
    non_elites = []
    for solution in solutions:
        if solution.solution_id not in elites:
            non_elites.append(solution)

    candidates = []

    # Sample candidates from both groups with 60% elite and 40% non-elite proportion
    # Total candidates: 5 total (3 from elites, 2 from non-elites)

    # Select 3 candidates from elites (60%)
    if len(elites) > 0:
        if len(elites) >= 3:
            elite_indices = np.random.choice(len(elites), size=3, replace=False)
            candidates.extend([elites[i] for i in elite_indices])
        else:
            # If fewer than 3 elites, use all available elites
            candidates.extend(elites)

    # Select 2 candidates from non-elites (40%)
    if len(non_elites) > 0:
        if len(non_elites) >= 2:
            non_elite_indices = np.random.choice(len(non_elites), size=2, replace=False)
            candidates.extend([non_elites[i] for i in non_elite_indices])
        else:
            # If fewer than 2 non-elites, use all available non-elites
            candidates.extend(non_elites)

    # If we have fewer than 5 candidates, fill the rest from the larger group
    if len(candidates) < 5:
        remaining_needed = 5 - len(candidates)
        if len(elites) > len(non_elites):
            # Prefer elite if elite group is larger
            available_elites = [e for e in elites if e not in candidates]
            if len(available_elites) > 0:
                if len(available_elites) >= remaining_needed:
                    elite_indices = np.random.choice(
                        len(available_elites), size=remaining_needed, replace=False
                    )
                    candidates.extend([available_elites[i] for i in elite_indices])
                else:
                    candidates.extend(available_elites)
        else:
            # Prefer non-elite if non-elite group is larger or equal
            available_non_elites = [ne for ne in non_elites if ne not in candidates]
            if len(available_non_elites) > 0:
                if len(available_non_elites) >= remaining_needed:
                    non_elite_indices = np.random.choice(
                        len(available_non_elites), size=remaining_needed, replace=False
                    )
                    candidates.extend(
                        [available_non_elites[i] for i in non_elite_indices]
                    )
                else:
                    candidates.extend(available_non_elites)

    # If still no candidates (both groups were empty), return None
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
