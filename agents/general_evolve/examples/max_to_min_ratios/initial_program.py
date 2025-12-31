"""Max to Min Ratios"""

# EVOLVE-BLOCK-START
import numpy as np
import scipy as sp


def optimize_construct(n=16, d=2):
    """
    Find n points in d-dimensional Euclidean space that minimize the ratio R = D_max / D_min,
    where D_max is the maximum Euclidean distance between any two points,
    and D_min is the minimum distance between any two distinct points in the set.
    # Constraints:
    1. All point coordinates must be real numbers.
    2. All points must be distinct.
    3. There are no bounds on coordinate ranges, but normalizing to set D_min = 1 is allowed.
    4. The search should favor diversity to avoid local minima.
    The smaller the value of ratix_squared, the better.

    Returns:
        points: A numpy array of shape (n, d).
            𝑛 points in the 𝑑-dimensional space so as to minimize the ratio between
            the maximum and minimum pairwise distances.
        ratio_squared: Ratio of max distance to min distance
    """

    # TODO:This is a return example that requires optimization to implement the complete algorithm
    points = np.zeros((n, d))
    ratio_squared = max_min_distance_ratio(points)
    return points, ratio_squared


def max_min_distance_ratio(points: np.ndarray):
    """Calculate the ratio of max distance to min distance"""
    pairwise_distances = sp.spatial.distance.pdist(points)
    min_distance = np.min(pairwise_distances)
    max_distance = np.max(pairwise_distances)
    if abs(min_distance) < 1e-10 or abs(max_distance) < 1e-10:
        return 0.0

    ratio_squared = (max_distance / min_distance) ** 2
    if ratio_squared is None or ratio_squared < 1e-10:
        return 0.0

    return ratio_squared


# EVOLVE-BLOCK-END

# This part remains fixed (not evolved)
if __name__ == "__main__":
    points, ratio_squared = optimize_construct(16, 2)
    print(f"Construction has {len(points)} points in {points.shape[1]} dimensions.")
    print(f"Ratio of max distance to min distance: sqrt({ratio_squared})")
