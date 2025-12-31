"""Initialize"""

import numpy as np


def generate_erdos_data() -> np.ndarray:
    """
    Generates half-segment optimization data for the Erdős minimum overlap problem
    using an adaptive random hill-climbing algorithm.

    Unlike the previous version, this function does not require fixed length or
    iteration arguments. It starts with a coarse sequence and progressively
    upsamples it (increases resolution) to refine the solution.

    Returns:
        best_half_seq: The optimized half-segment sequence.
    """
    # --- Configuration for Adaptive Strategy ---
    initial_n = 50  # Start with a small size for fast initial convergence
    max_n = 400  # Maximum resolution limit (prevent infinite growth)
    patience = 2000  # Stop current stage if no improvement for this many iters
    upsample_factor = 2  # Double the size at each stage

    # 1. Initialization
    current_n = initial_n
    # Start with 0.5 + noise
    current_half = np.ones(current_n) * 0.5 + np.random.normal(0, 0.01, current_n)

    best_global_loss = float("inf")
    best_half_global = None

    print("Starting Adaptive Optimization...")

    while current_n <= max_n:
        # Calculate target sum for current length
        # full_len = 2*N - 1
        full_len = 2 * current_n - 1
        target_sum = full_len / 2.0

        # --- Helper Functions (defined inside loop to use current context) ---
        def get_full_sequence(half_seq):
            reversed_seq = half_seq[::-1]
            return np.concatenate((half_seq[:-1], reversed_seq))

        def enforce_constraints(half_seq):
            # Clip to [0, 1]
            half_seq = np.clip(half_seq, 0, 1)
            # Fix sum
            for _ in range(3):
                current_full_sum = 2 * np.sum(half_seq) - half_seq[-1]
                if current_full_sum == 0:
                    break
                scale_factor = target_sum / current_full_sum
                half_seq = half_seq * scale_factor
                half_seq = np.clip(half_seq, 0, 1)
            return half_seq

        def compute_loss(half_seq):
            full_seq = get_full_sequence(half_seq)
            cross_corr = np.correlate(full_seq, 1 - full_seq, mode="full")
            max_overlap = np.max(cross_corr) / len(full_seq) * 2
            return max_overlap

        # Initial constraint check for this stage
        current_half = enforce_constraints(current_half)
        stage_best_loss = compute_loss(current_half)
        stage_best_half = current_half.copy()

        print(f"\n--- Stage: Length N={current_n} ---")

        # Optimization Loop for this stage
        no_improve_counter = 0
        iter_count = 0

        while no_improve_counter < patience:
            iter_count += 1

            # Mutation
            mutation = stage_best_half.copy()
            # Adaptive mutation rate: fewer points as N grows
            num_points = max(1, int(current_n * 0.05))
            idx_to_change = np.random.randint(0, current_n, size=num_points)
            noise = np.random.normal(0, 0.1, size=num_points)
            mutation[idx_to_change] += noise

            mutation = enforce_constraints(mutation)
            new_loss = compute_loss(mutation)

            if new_loss < stage_best_loss:
                stage_best_loss = new_loss
                stage_best_half = mutation.copy()
                no_improve_counter = 0  # Reset counter
                # Update global best if applicable
                if stage_best_loss < best_global_loss:
                    best_global_loss = stage_best_loss
                    best_half_global = stage_best_half.copy()
            else:
                no_improve_counter += 1

            if iter_count % 2000 == 0:
                print(
                    f"  Iter {iter_count}: Loss={stage_best_loss:.6f} (No improve for {no_improve_counter})"
                )

        print(f"  Stage finished. Best Loss for N={current_n}: {stage_best_loss:.6f}")

        # Prepare for next stage: Upsample (Interpolate)
        if current_n * upsample_factor <= max_n:
            new_n = current_n * upsample_factor
            # Linear interpolation to expand the array
            x_old = np.linspace(0, 1, current_n)
            x_new = np.linspace(0, 1, new_n)
            current_half = np.interp(x_new, x_old, stage_best_half)
            current_n = new_n
        else:
            break  # Stop if next size exceeds max_n

    print(
        f"Optimization complete. Final Length = {len(best_half_global)}, Final Overlap = {best_global_loss:.6f}"
    )
    return best_half_global


def verify_sequence(sequence: list[float]):
    """Raises an error if the sequence does not satisfy the constraints."""
    # Check that all values are between 0 and 1.
    if not all(0 <= val <= 1 for val in sequence):
        raise AssertionError("All values in the sequence must be between 0 and 1.")
    # Check that the sum of values in the sequence is exactly n / 2.0.
    # Note: Due to floating point errors in adaptive scaling, we use np.isclose
    target_val = len(sequence) / 2.0
    current_sum = np.sum(sequence)
    if not np.isclose(current_sum, target_val, rtol=1e-5):
        raise AssertionError(
            "The sum of values in the sequence must be exactly n / 2.0. The sum is "
            f"{current_sum} but it should be {target_val}."
        )
    print(
        "The sequence generates a valid step function for Erdős' minimum "
        "overlap problem."
    )


def compute_upper_bound(sequence: list[float]) -> float:
    """Returns the upper bound for a sequence of coefficients."""
    convolution_values = np.correlate(
        np.array(sequence), 1 - np.array(sequence), mode="full"
    )
    return np.max(convolution_values) / len(sequence) * 2


if __name__ == "__main__":
    # 1. Get data (Arguments are removed) You generated code
    best_sequence = generate_erdos_data()

    # 2. Execute your required external operations
    reversed_sequence = best_sequence[::-1]
    final_sequence = np.concatenate((best_sequence[:-1], reversed_sequence))

    # 3. Verify and Print
    verify_sequence(final_sequence)
    print("Upper Bound:", compute_upper_bound(final_sequence))
