# EVOLVE-BLOCK-START
"""Constructor-based heilbronn problem for triangles for n=11 points"""

import math
import random

SQRT3 = math.sqrt(3.0)
HALF_SQRT3 = SQRT3 / 2.0
TOL = 1e-6
TOL_SQ = TOL * TOL


def halton(index, base):
    """Generate Halton sequence number for given index and base."""
    result = 0.0
    f = 1.0
    i = index
    while i > 0:
        i, remainder = divmod(i, base)
        result += f * remainder
        f /= base
    return result


def adaptive_hybrid_optimization_one_run(n, max_iter=2000, k=50, decay=0.95):
    """
    Optimize point placement to maximize the minimal triangle area (single run).

    Parameters:
        n: Number of points.
        max_iter: Maximum iterations for optimization.
        k: Number of perturbation trials per point.
        decay: Decay factor for perturbation size.

    Returns:
        List of optimized points.
    """
    points = [(0.0, 0.0), (1.0, 0.0), (0.5, HALF_SQRT3)]
    if n <= 3:
        return points[:n]

    halton_index = random.randint(0, 1000000)
    for i in range(3, n):
        while True:
            u = halton(halton_index, 2)
            v = halton(halton_index, 3)
            halton_index += 1
            if u + v > 1.0:
                u = 1.0 - u
                v = 1.0 - v
            x = u + v * 0.5
            y = v * HALF_SQRT3 * 2.0 / SQRT3 * SQRT3 / 2.0
            if point_in_triangle(x, y) and not any(
                points_are_close((x, y), p) for p in points
            ):
                points.append((x, y))
                break

    global_min = min_triangle_area(points)
    delta = 0.1
    consecutive_failures = 0

    for iter in range(max_iter):
        idx = random.randint(3, n - 1)
        rest_points = points[:idx] + points[idx + 1 :]
        non_moved_min = min_triangle_area(rest_points)
        best_move = None
        best_new_global_min = global_min

        for trial in range(k):
            dx = random.uniform(-delta, delta)
            dy = random.uniform(-delta, delta)
            new_x = points[idx][0] + dx
            new_y = points[idx][1] + dy
            new_point = (new_x, new_y)

            if not point_in_triangle(new_x, new_y):
                continue

            too_close = False
            for p in rest_points:
                if points_are_close(new_point, p):
                    too_close = True
                    break
            if too_close:
                continue

            candidate_global_min = non_moved_min
            m = len(rest_points)
            break_outer = False
            for i in range(m):
                for j in range(i + 1, m):
                    area = triangle_area(rest_points[i], rest_points[j], new_point)
                    if area < candidate_global_min:
                        candidate_global_min = area
                    if candidate_global_min < best_new_global_min:
                        break_outer = True
                        break
                if break_outer:
                    break

            if candidate_global_min >= best_new_global_min:
                best_new_global_min = candidate_global_min
                best_move = new_point

        if best_move is not None:
            points[idx] = best_move
            global_min = best_new_global_min
            consecutive_failures = 0
        else:
            consecutive_failures += 1

        if consecutive_failures >= 10:
            delta *= decay
            consecutive_failures = 0
            if delta < 1e-6:
                break
    return points


def find_best_placement(n, max_iter=2000, k=50, decay=0.95, num_runs=5):
    """
    Optimize point placement using multiple runs to escape local optima.

    Parameters:
        n: Number of points.
        max_iter: Maximum iterations per run.
        k: Number of perturbation trials per point per run.
        decay: Decay factor for perturbation size per run.
        num_runs: Number of independent optimization runs.

    Returns:
        Best set of points found across all runs.

    """
    best_points = None
    best_min_area = 0.0

    for run in range(num_runs):
        points = adaptive_hybrid_optimization_one_run(n, max_iter, k, decay)
        min_area_val = min_triangle_area(points)
        if min_area_val > best_min_area:
            best_min_area = min_area_val
            best_points = points

    min_area = best_min_area / triangle_area((0.0, 0.0), (1.0, 0.0), (0.5, HALF_SQRT3))
    return best_points, min_area


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


def point_in_triangle(x, y):
    """Check if a point (x, y) lies within the equilateral triangle."""
    return (y >= 0) and (y <= SQRT3 * min(x, 1 - x)) and (SQRT3 * x <= SQRT3 - y)


def points_are_close(p1, p2, tol=TOL):
    """Check if two points are closer than a given tolerance."""
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) < tol


# EVOLVE-BLOCK-END


# This part remains fixed (not evolved)
def run_search_point(n=11):
    """
    Run a search point for a given number of iterations.
    """
    points, min_area = find_best_placement(n)

    return points, min_area


if __name__ == "__main__":
    """Run the heilbronn problem for triangles constructor for n=11"""
    n = 11  # Number of points
    optimized_points, min_area = run_search_point(n)
    for i, p in enumerate(optimized_points):
        print(f"P{i + 1}: ({p[0]:.6f}, {p[1]:.6f})")
    print(f"Minimal Triangle Area: {min_area:.8f}")
