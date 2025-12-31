"""Best result evaluation for sums and differences problems 1."""
best_list = [15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
             31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46,
             47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62,
             63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78,
             79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94,
             95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108,
             109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121,
             122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134,
             135, 136, 137, 138, 139, 140, 141, 143, 144, 145, 146, 147, 148,
             150, 151, 179, 196, 198, 202, 211, 213, 222, 227, 228, 229, 233,
             241, 249, 255, 257, 260, 264, 266, 267, 269, 270, -104]

import math

from numba import njit


def get_score(best_list):
  """Returns the score for the given list using Numba."""
  if len(best_list) < 2:
    return 0

  # if the list contains non-integers, return 0
  if not all(isinstance(x, int) for x in best_list):
    return 0

  return get_score_numba(best_list) + (1.0 - 1.0 / len(set(best_list))) / 100.0
  # return get_score_numba(best_list)


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


print(f"Final score: {get_score(best_list)}")