"""
Evolux: Heilbronn problem for convex regions (n=13) Best Result
"""

import os
import sys
import numpy as np
from scipy.spatial import ConvexHull

# --- Constants ---
TOL = 1e-6


def validate_placement(points: np.ndarray):
    """Checks that all points are inside the unit square [0,1]x[0,1],
    that no points overlap, and that any three points can form a triangle."""

    def is_collinear(p1, p2, p3):
        # Using a tolerance for floating point comparisons
        return (
            abs(
                p1[0] * (p2[1] - p3[1])
                + p2[0] * (p3[1] - p1[1])
                + p3[0] * (p1[1] - p2[1])
            )
            < TOL
        )

    # Check for duplicates by checking distance between all pairs of points
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            if np.linalg.norm(points[i] - points[j]) < TOL:
                return False, f"Duplicate points found: P{i} and P{j} are too close."

    # Check if points are inside the unit square
    for i, (x, y) in enumerate(points):
        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
            return False, f"Point P{i} ({x:.4f}, {y:.4f}) is outside the unit square."

    # Check if any three points are collinear
    num_points = len(points)
    for i in range(num_points):
        for j in range(i + 1, num_points):
            for k in range(j + 1, num_points):
                if is_collinear(points[i], points[j], points[k]):
                    return False, f"Points P{i}, P{j}, P{k} are collinear."

    return True, None


def triangle_area(a, b, c):
    """Calculate the area of a triangle given three vertices using the cross product formula."""
    return 0.5 * abs(a[0] * (b[1] - c[1]) + b[0] * (c[1] - a[1]) + c[0] * (a[1] - b[1]))


def min_triangle_area(points):
    """Calculate the minimal triangle area formed by any three points in the set."""
    min_area = float("inf")
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                area = triangle_area(points[i], points[j], points[k])
                if area < min_area:
                    min_area = area
    return min_area


# Best points
found_points = np.array(
    [
        (0.3591925621199088, 0.3334510822264365),
        (0.7151172951637265, 0.34817605005105307),
        (0.03245319952979969, 0.8544134306573824),
        (0.10204795506287506, 0.18775355475689087),
        (0.3809914687592066, 0.0),
        (0.5020135091515034, 1.0),
        (1.0, 0.5011436505878603),
        (0.5171813243193137, 0.7278491057489135),
        (0.980631493052252, 0.22378415615620256),
        (0.0, 0.474315845988681),
        (0.17143850828909613, 1.0),
        (0.841940674338323, 0.7939733168630484),
        (0.7390238137041651, 0.04477290011742902),
    ]
)


if __name__ == "__main__":
    valid, reason = validate_placement(found_points)
    if not valid:
        print(f"validate failed. err:{reason}")
        sys.exit(1)

    print(f"validate result.valid:{valid}")

    min_area = min_triangle_area(found_points)
    convex_hull_area = ConvexHull(found_points).volume
    min_area_normalized = min_area / convex_hull_area

    print(f"Minimum triangle area: {min_area_normalized:.16f}")
