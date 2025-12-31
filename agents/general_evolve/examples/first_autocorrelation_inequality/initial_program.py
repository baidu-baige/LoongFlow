# EVOLVE-BLOCK-START
"""First Autocorrelation inequality problem."""

import numpy as np


def search_for_best_sequence():
    """Generate optimized step function using refined boundary-focused approach"""
    dimension = 600
    interval_start = -1 / 4
    interval_end = 1 / 4

    # Create symmetric step function with higher values near boundaries
    x = np.linspace(interval_start, interval_end, dimension)

    # Use a combination of quadratic and linear terms to emphasize boundaries
    step_function = 1.0 + 4.0 * np.abs(x) - 16.0 * x**2

    # Enforce perfect symmetry
    step_function = (step_function + step_function[::-1]) / 2

    # Ensure all values are positive
    step_function = np.maximum(step_function, 0)

    # Normalize to have total mass 1 (for consistent comparison)
    step_function /= np.sum(step_function)

    return step_function


# EVOLVE-BLOCK-END


# The following part remains fixed (not evolved)
def evaluate_sequence(sequence: list[float]) -> float:
    """
    Evaluates a sequence of coefficients with enhanced security checks.
    Returns np.inf if the input is invalid.
    """
    # --- Security Checks ---

    # Verify that the input is a list
    if not isinstance(sequence, list):
        return np.inf

    # Reject empty lists
    if not sequence:
        return np.inf

    # Check each element in the list for validity
    for x in sequence:
        # Reject boolean types (as they are a subclass of int) and
        # any other non-integer/non-float types (like strings or complex numbers).
        if isinstance(x, bool) or not isinstance(x, (int, float)):
            return np.inf

        # Reject Not-a-Number (NaN) and infinity values.
        if np.isnan(x) or np.isinf(x):
            return np.inf

    # Convert all elements to float for consistency
    sequence = [float(x) for x in sequence]

    # Protect against negative numbers
    sequence = [max(0, x) for x in sequence]

    # Protect against numbers that are too large
    sequence = [min(1000.0, x) for x in sequence]

    n = len(sequence)
    b_sequence = np.convolve(sequence, sequence)
    max_b = max(b_sequence)
    sum_a = np.sum(sequence)

    # Protect against the case where the sum is too close to zero
    if sum_a < 0.01:
        return np.inf

    return float(2 * n * max_b / (sum_a**2))


def run_search_for_best_sequence():
    """
    Run the search for the best sequence
    """
    best_sequence = search_for_best_sequence()

    return best_sequence


if __name__ == "__main__":
    """Run the first_autocorrelation_inequality problem for best_sequence"""
    best_sequence = run_search_for_best_sequence()
    print(f"Best sequence: {best_sequence}")
    for i, p in enumerate(best_sequence):
        print(f"P{i + 1}: ({p:.6f})")
