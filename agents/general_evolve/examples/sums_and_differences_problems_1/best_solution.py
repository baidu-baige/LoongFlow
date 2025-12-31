""" "Best solution search for sums and differences problems 1."""

import math
import random
from typing import Tuple, List

import numpy as np
from numba import njit


def next_prime(n: int) -> int:
    """Returns the next prime number greater than or equal to n."""
    if n < 2:
        return 2
    candidate = n + (1 if n % 2 == 0 else 0)
    while True:
        if candidate % 2 == 0:
            candidate += 1
            continue
        is_prime = True
        if candidate == 2:
            return candidate
        i = 3
        while i * i <= candidate:
            if candidate % i == 0:
                is_prime = False
                break
            i += 2
        if is_prime:
            return candidate
        candidate += 2


def generate_cubic_residues(n: int, p: int) -> List[int]:
    """Generates the first n cubic residues modulo p."""
    residues = set()
    for i in range(1, p):
        residues.add(pow(i, 3, p))
    residues = sorted(residues)
    return residues[:n]


def probabilistic_sidon_verify(A: List[int], tolerance: float = 0.001) -> bool:
    """Probabilistic verification of Sidon set property."""
    n = len(A)
    if n == 0:
        return True
    max_observed = 2 * max(A)
    check_array = [False] * (max_observed + 1)
    collisions = 0
    total_pairs = 0

    for i in range(n):
        for j in range(i, n):
            s = A[i] + A[j]
            total_pairs += 1
            if s <= max_observed:
                if check_array[s]:
                    collisions += 1
                    if collisions / total_pairs > tolerance:
                        return False
                else:
                    check_array[s] = True
    return collisions / total_pairs <= tolerance


def generate_projective_set(q: int) -> List[int]:
    """Generates the projective plane set for a given q."""
    affine_points = []
    for x in range(q):
        for y in range(q):
            affine_points.append(x * q + y)
    infinity_points = [q * q + x for x in range(q)]
    infinity_points.append(q * q + q)
    return affine_points + infinity_points


def boundary_annealing(
    input_set: List[int],
    max_iter: int = 10000,
    T0: float = 1.0,
    alpha: float = 0.995,
    target_score: float = 1.1319,
) -> Tuple[List[int], float]:
    """Performs boundary-focused simulated annealing to optimize the set."""
    current_set = input_set[:]
    n = len(current_set)
    boundary_size = max(1, n // 10)
    left_boundary_indices = list(range(boundary_size))
    right_boundary_indices = list(range(n - boundary_size, n))
    current_score = get_score(current_set)
    best_set = current_set[:]
    best_score = current_score
    T = T0
    for iter in range(max_iter):
        new_set = current_set[:]
        if random.random() < 0.5:
            boundary = random.choice(["left", "right"])
            if boundary == "left":
                indices = left_boundary_indices
            else:
                indices = right_boundary_indices
            if random.random() < 0.5:
                idx = random.choice(indices)
                old_val = new_set[idx]
                delta = random.randint(-10, 10)
                new_val = old_val + delta
                if new_val in new_set:
                    new_val = old_val
                else:
                    new_set[idx] = new_val
            else:
                i1, i2 = random.sample(indices, 2)
                new_set[i1], new_set[i2] = new_set[i2], new_set[i1]
        else:
            idx = random.choice(left_boundary_indices + right_boundary_indices)
            old_val = new_set[idx]
            delta = random.randint(-10, 10)
            new_val = old_val + delta
            if new_val in new_set:
                new_val = old_val
            else:
                new_set[idx] = new_val
        new_score = get_score(new_set)
        if new_score >= target_score:
            best_set = new_set
            best_score = new_score
            current_set = new_set
            break
        if new_score > current_score or random.random() < math.exp(
            (new_score - current_score) / T
        ):
            current_set = new_set
            current_score = new_score
            if new_score > best_score:
                best_set = new_set
                best_score = new_score
        T *= alpha
    return best_set, best_score


def search_for_best_set() -> Tuple[np.ndarray, str]:
    """Searches for the best set by fusing projective and cubic residue sets."""
    q = random.randint(11, 20)
    n_projective = q * q + q + 1
    projective_set = generate_projective_set(q)

    p_base = random.randint(300, 1000)
    p_cubic = next_prime(p_base)
    n_cubic_gen = random.randint(150, 300)
    cubic_set = generate_cubic_residues(n_cubic_gen, p_cubic)

    projective_segments = []
    segment_size = random.randint(8, 15)

    n_segments_proj = len(projective_set) // segment_size

    for i in range(n_segments_proj):
        start = i * segment_size
        end = start + segment_size
        segment = projective_set[start:end]
        projective_segments.append(segment)

    cubic_segments = []
    n_segments_cubic = len(cubic_set) // segment_size

    for i in range(n_segments_cubic):
        start = i * segment_size
        end = start + segment_size
        segment = cubic_set[start:end]
        cubic_segments.append(segment)

    min_jaccard = float("inf")
    best_i = -1
    best_j = -1

    for i in range(n_segments_proj):
        for j in range(n_segments_cubic):
            set1 = set(projective_segments[i])
            set2 = set(cubic_segments[j])
            intersection = len(set1 & set2)
            union = len(set1 | set2)
            jaccard = intersection / union if union > 0 else float("inf")
            if jaccard < min_jaccard:
                min_jaccard = jaccard
                best_i = i
                best_j = j

    if best_i == -1:
        best_i, best_j = 0, 0

    segment_to_replace = projective_segments[best_i]
    cubic_segment = cubic_segments[best_j]

    min_val_projective = min(projective_set)
    max_val_projective = max(projective_set)
    min_val_cubic = min(cubic_segment) if cubic_segment else 0

    shift = max_val_projective - min_val_cubic + 1
    shifted_cubic_segment = [x + shift for x in cubic_segment]

    fused_set = (
        projective_set[: best_i * segment_size]
        + shifted_cubic_segment
        + projective_set[best_i * segment_size + segment_size :]
    )

    rand_T0 = random.uniform(0.5, 2.5)
    rand_alpha = random.uniform(0.990, 0.999)
    rand_iter = random.randint(8000, 15000)

    annealed_fused_set, fused_score = boundary_annealing(
        fused_set, max_iter=rand_iter, T0=rand_T0, alpha=rand_alpha
    )
    annealed_fused_set = list(set(annealed_fused_set))

    if len(annealed_fused_set) != n_projective:
        annealed_fused_set = annealed_fused_set[:n_projective]

    fused_verified = probabilistic_sidon_verify(annealed_fused_set, tolerance=0.0005)

    if fused_verified:
        return (
            np.array(annealed_fused_set),
            f"Projective-Cubic fused (q={q}, p={p_cubic}, score={fused_score:.4f})",
        )

    pure_projective_set = projective_set[:]
    annealed_pure_set, pure_score = boundary_annealing(
        pure_projective_set, max_iter=rand_iter, T0=rand_T0, alpha=rand_alpha
    )
    pure_verified = probabilistic_sidon_verify(annealed_pure_set, tolerance=0.0005)

    if pure_verified:
        return (
            np.array(annealed_pure_set),
            f"Pure projective (q={q}, score={pure_score:.4f})",
        )

    pure_set_score = get_score(pure_projective_set)
    if pure_set_score > fused_score:
        return (
            np.array(pure_projective_set),
            f"Pure projective without annealing (q={q}, score={pure_set_score:.4f})",
        )
    else:
        return (
            np.array(annealed_fused_set),
            f"Projective-Cubic fused without verification (q={q}, p={p_cubic}, score={fused_score:.4f})",
        )


def get_score(best_list):
    """Returns the score for the given list using Numba."""
    if isinstance(best_list, np.ndarray):
        best_list = best_list.tolist()

    try:
        best_list = [int(x) for x in best_list]
    except (ValueError, TypeError):
        print(f"get_score List contains non-convertible values: {best_list}")
        return 0

    if len(best_list) < 2:
        print(f"get_score List is too short: {best_list}")
        return 0

    # if the list contains non-integers, return 0
    if not all(isinstance(x, int) for x in best_list):
        print(f"get_score List contains non-integers: {best_list}")
        return 0

    return get_score_numba(best_list) + (1.0 - 1.0 / len(set(best_list))) / 100.0


@njit
def get_score_numba(best_list):
    """Returns the score for the given list using Numba."""

    best_list_set = set(best_list)
    n_unique = len(best_list_set)

    a_minus_a = set()
    for i in best_list:
        for j in best_list:
            a_minus_a.add(i - j)

    a_plus_a = set()
    for i in best_list:
        for j in best_list:
            a_plus_a.add(i + j)

    lhs = len(a_minus_a) / n_unique
    rhs = len(a_plus_a) / n_unique

    try:
        return math.log(rhs) / math.log(lhs)
    except Exception:
        return 0


if __name__ == "__main__":
    """Main optimization loop to find the best set."""
    NUM_ITERATIONS = 100

    global_best_score = -1.0
    global_best_set = None
    global_best_desc = ""

    for i in range(NUM_ITERATIONS):
        print(f"\n====== Iteration {i + 1:02d} / {NUM_ITERATIONS} ======")
        current_set_np, current_desc = search_for_best_set()
        current_set_list = current_set_np.tolist()

        current_score = get_score(current_set_list)

        print(
            f"Iter {i+1:02d}: Score = {current_score:.6f} | Size = {len(current_set_list)} | {current_desc}"
        )

        if current_score > global_best_score:
            global_best_score = current_score
            global_best_set = current_set_np
            global_best_desc = current_desc
            print(f"   >>> NEW BEST FOUND: {global_best_score:.18f} <<<")
            print(f"      Size: {len(current_set_list)}")
            print(f"      global_best_set: {global_best_set.tolist()}")

    print("\n" + "=" * 30)
    print("FINAL SEARCH RESULTS")
    print("=" * 30)

    if global_best_set is not None:
        final_list = sorted(global_best_set.tolist())
        print(f"Best Score Found: {global_best_score}")
        print(f"Description: {global_best_desc}")
        print(f"Set Size: {len(final_list)}")
        print(f"Elements: {final_list}")
    else:
        print("No valid set found.")
