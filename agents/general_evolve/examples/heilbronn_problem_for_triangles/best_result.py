"""
Evolux: Heilbronn problem for triangles (n=11) Best Result
"""

import numpy as np
import sys


def validate_placement(points: np.ndarray):
    """Checks that all points are inside the triangle with vertices (0,0), (1,0), (0.5, sqrt(3)/2),
    that no points overlap, and that any three points can form a triangle."""

    def is_collinear(p1, p2, p3):
        # Calculate the cross product of vectors (p2 - p1) and (p3 - p1)
        return (p2[0] - p1[0]) * (p3[1] - p1[1]) == (p3[0] - p1[0]) * (p2[1] - p1[1])

    # Check for duplicates by converting points to a set of tuples
    unique_points = set(map(tuple, points))
    if len(unique_points) != len(points):
        return (
            False,
            "Duplicate points found.",
        )  # Return False if there are overlapping points

    # Check if points are inside the triangle
    for x, y in points:
        if not (
            (y >= 0) and (np.sqrt(3) * x <= np.sqrt(3) - y) and (y <= np.sqrt(3) * x)
        ):
            return False, f"Point ({x:.4f}, {y:.4f}) is outside the triangle."

    # Check if any three points can form a triangle (i.e., are not collinear)
    num_points = len(points)
    for i in range(num_points):
        for j in range(i + 1, num_points):
            for k in range(j + 1, num_points):
                if is_collinear(points[i], points[j], points[k]):
                    return False, "Three or more points are collinear."

    return True, None


def triangle_area(a, b, c):
    """Calculate the area of a triangle given three vertices using the cross product formula."""
    return abs(a[0] * (b[1] - c[1]) + b[0] * (c[1] - a[1]) + c[0] * (a[1] - b[1])) / 2.0


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
        [0.4279845295543236, 0.7412909500398202],
        [0.1062310826141215, 0.0710766927753885],
        [0.5906647921426612, 0.4392918971067320],
        [0.8521744270583328, 0.2560414029912148],
        [0.7225470220844938, 0.0000000000010000],
        [0.5720154704456765, 0.7412909500398202],
        [0.8937689173858787, 0.0710766927753883],
        [0.4093352078573388, 0.4392918971067318],
        [0.1478255729416673, 0.2560414029912148],
        [0.2774529779155065, 0.0000000000010000],
        [0.5000000000000000, 0.2111264259190374],
    ]
)

# Vertices of an equilateral triangle that contains the points.
a = np.array([0, 0])
b = np.array([1, 0])
c = np.array([0.5, np.sqrt(3) / 2])

if __name__ == "__main__":
    valid, reason = validate_placement(found_points)
    if not valid:
        print(f"validate failed. err:{reason}")
        sys.exit(1)

    print(f"validate result.valid:{valid}")

    min_area = min_triangle_area(found_points)
    min_area_normalized = min_area / triangle_area(a, b, c)

    print(f"Minimum triangle area: {min_area_normalized:.16f}")
