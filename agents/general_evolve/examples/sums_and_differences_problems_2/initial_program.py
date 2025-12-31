""" Init program """

import math
import time
from typing import Tuple

import numpy as np
from scipy import optimize

minimize = optimize.minimize


def search_for_best_set() -> Tuple[np.ndarray, str]:
    """Searches for joint probability distributions maximizing the ratio."""
    # You should modify best_list_iqhg init value
    best_list_iqhg = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    best_list = best_list_iqhg.copy()
    curr_list = best_list.copy()
    best_score = get_score(best_list)
    eval_count = 0
    start_time = time.time()
    while time.time() - start_time < 1000:  # Search for 1000 seconds
        # Mutate best construction
        random_index = np.random.randint(0, len(curr_list))
        curr_list[random_index] += np.random.randint(-3, 4)
        score = get_score(curr_list)
        eval_count += 1
        if np.random.rand() < 0.05:
            curr_list.append(np.random.randint(1, 20))
        if np.random.rand() < 0.03 and len(curr_list) > 5:
            curr_list = curr_list[1:]
        if len(curr_list) > 20:
            curr_list = curr_list[10:]
        if score > best_score:
            best_score = score
            best_list = curr_list.copy()
            print(f'Best score: {score}')
            print(set(best_list))
    return best_list


def get_score(best_list):
    """Returns the score for the given list using Numba."""
    if len(best_list) < 2:
        return 0

    for i in best_list:
        if i > 2_000_000_000:
            return 0

    # Take the absolute value of the list
    best_list = [abs(x) for x in best_list]

    # if the list contains non-integers, return 0
    if not all(isinstance(x, int) for x in best_list):
        return 0

    return get_score_numba(best_list) + (1.0 - 1.0 / len(set(best_list))) / 100.0

from numba import njit
@njit
def get_score_numba(best_list):
    """Returns the score for the given list using Numba."""

    best_list_set = set(best_list)
    # Add 0 to the set
    best_list_set.add(0)
    n_unique = len(best_list_set)

    a_minus_a = set()
    for i in best_list:
        for j in best_list:
            a_minus_a.add(i - j)

    a_plus_a = set()
    for i in best_list:
        for j in best_list:
            a_plus_a.add(i + j)

    lhs = len(a_minus_a)
    rhs = len(a_plus_a)

    denominator = math.log(lhs / rhs)

    # Find the maximum value of best_list_set
    max_value = max(best_list_set)

    try:
        return denominator / math.log(2 * max_value + 1) + 1.0
    except Exception:
        return 0