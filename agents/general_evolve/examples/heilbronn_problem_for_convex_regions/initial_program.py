# EVOLVE-BLOCK-START
"""Constructor-based heilbronn problem for convex regions"""

import math
import random

TOL = 1e-6
TOL_SQ = TOL * TOL
REGION_AREA = 1.0  # Unit square area


def points_are_close(p1, p2):
    """Check if two points are closer than a tolerance threshold."""
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    return dx * dx + dy * dy < TOL_SQ


def triangle_area(a, b, c):
    """Calculate area of triangle given three vertices using cross product."""
    return 0.5 * abs((b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1]))


def initial_placement_13():
    """Generate hexagonal lattice initialization for 13 points in unit square."""
    margin = 0.05
    W = 1 - 2 * margin
    H = 1 - 2 * margin
    points = []

    # Three rows: bottom (4), middle (5), top (4)
    dy = H / 2.0
    dx = W / 4.0

    # Bottom row (4 points)
    y_bottom = margin
    for j in range(4):
        x = margin + dx / 2 + j * dx
        points.append((x, y_bottom))

    # Middle row (5 points)
    y_middle = margin + dy
    for j in range(5):
        x = margin + j * dx
        points.append((x, y_middle))

    # Top row (4 points)
    y_top = margin + 2 * dy
    for j in range(4):
        x = margin + dx / 2 + j * dx
        points.append((x, y_top))

    return points


def random_initial_placement(n):
    """Random initialization with collision avoidance."""
    points = []
    while len(points) < n:
        x = random.uniform(0, 1)
        y = random.uniform(0, 1)
        new_point = (x, y)
        too_close = False
        for p in points:
            if points_are_close(new_point, p):
                too_close = True
                break
        if not too_close:
            points.append(new_point)
    return points


def initial_placement(n):
    """Select initialization strategy based on n."""
    if n == 13:
        return initial_placement_13()
    return random_initial_placement(n)


def precompute_triangles(points):
    """Precompute all triangles and point-triangle mappings."""
    n = len(points)
    triangles = []  # List of (i, j, k) triplets
    areas = []  # Corresponding triangle areas
    point_triangles = [[] for _ in range(n)]  # Triangles per point

    tri_idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                triangles.append((i, j, k))
                area_val = triangle_area(points[i], points[j], points[k])
                areas.append(area_val)
                point_triangles[i].append(tri_idx)
                point_triangles[j].append(tri_idx)
                point_triangles[k].append(tri_idx)
                tri_idx += 1
    return triangles, areas, point_triangles


def find_best_placement(n=13, max_iter=200000, T0=0.05, decay=0.9999, num_runs=10):
    """Optimized Heilbronn solver using simulated annealing with adaptive mechanisms."""
    best_overall_min_area = 0.0
    best_overall_points = []

    for run in range(num_runs):
        points = initial_placement(n)
        triangles, areas, point_triangles = precompute_triangles(points)
        current_min_area = min(areas) if areas else 0.0
        best_run_points = [p for p in points]
        best_run_min_area = current_min_area
        current_energy = -current_min_area
        T = T0
        max_step = 0.05
        accepted_count = 0
        total_count = 0
        non_improvement_count = 0

        for iter in range(max_iter):
            idx = random.randint(0, n - 1)
            old_point = points[idx]
            dx = random.uniform(-max_step, max_step)
            dy = random.uniform(-max_step, max_step)
            new_x = max(0.0, min(1.0, old_point[0] + dx))
            new_y = max(0.0, min(1.0, old_point[1] + dy))
            new_point = (new_x, new_y)

            # Check if new point is too close to others
            too_close = False
            for i, p in enumerate(points):
                if i == idx:
                    continue
                if points_are_close(new_point, p):
                    too_close = True
                    break
            if too_close:
                T = T0 * (decay**iter)
                non_improvement_count += 1
                continue

            # Backup affected areas
            backup_areas = [areas[tri_idx] for tri_idx in point_triangles[idx]]
            old_min_area = current_min_area

            # Update point and recompute affected areas
            points[idx] = new_point
            min_changed_area = float("inf")
            for tri_idx in point_triangles[idx]:
                i, j, k = triangles[tri_idx]
                area_val = triangle_area(points[i], points[j], points[k])
                areas[tri_idx] = area_val
                if area_val < min_changed_area:
                    min_changed_area = area_val

            # Skip degenerate cases
            if min_changed_area < 1e-10:
                points[idx] = old_point
                for tri_idx, area_val in zip(point_triangles[idx], backup_areas):
                    areas[tri_idx] = area_val
                T = T0 * (decay**iter)
                non_improvement_count += 1
                continue

            # Compute new global minimum area
            current_min_area = min(areas)
            new_energy = -current_min_area
            delta_energy = new_energy - current_energy

            # Acceptance criteria
            if delta_energy < 0 or random.random() < math.exp(-delta_energy / T):
                accepted_count += 1
                current_energy = new_energy
                if current_min_area > best_run_min_area:
                    best_run_min_area = current_min_area
                    best_run_points = [p for p in points]
                    non_improvement_count = 0
                else:
                    non_improvement_count += 1
            else:
                points[idx] = old_point
                for tri_idx, area_val in zip(point_triangles[idx], backup_areas):
                    areas[tri_idx] = area_val
                current_min_area = old_min_area
                non_improvement_count += 1

            total_count += 1
            T = T0 * (decay**iter)

            # Adaptive step size adjustment
            if iter % 100 == 0:
                if total_count > 0:
                    accept_rate = accepted_count / 100.0
                    if accept_rate < 0.4:
                        max_step *= 0.9
                    elif accept_rate > 0.6:
                        max_step *= 1.1
                    max_step = max(0.001, min(max_step, 0.1))
                    accepted_count = 0
                    total_count = 0

            # Big kick to escape local minima
            if non_improvement_count >= 2000:
                candidate_points = []
                for i in range(n):
                    found = False
                    for _ in range(5):
                        dx_kick = random.uniform(-max_step, max_step)
                        dy_kick = random.uniform(-max_step, max_step)
                        new_x = points[i][0] + dx_kick
                        new_y = points[i][1] + dy_kick
                        new_p = (max(0.0, min(1.0, new_x)), max(0.0, min(1.0, new_y)))

                        too_close = False
                        for j in range(len(candidate_points)):
                            if points_are_close(new_p, candidate_points[j]):
                                too_close = True
                                break
                        if too_close:
                            continue

                        candidate_points.append(new_p)
                        found = True
                        break

                    if not found:
                        candidate_points.append(points[i])

                points = candidate_points
                for tri_idx in range(len(triangles)):
                    i, j, k = triangles[tri_idx]
                    areas[tri_idx] = triangle_area(points[i], points[j], points[k])
                current_min_area = min(areas)
                current_energy = -current_min_area
                if current_min_area > best_run_min_area:
                    best_run_min_area = current_min_area
                    best_run_points = [p for p in points]
                non_improvement_count = 0

            if T < 1e-8:
                break

        if best_run_min_area > best_overall_min_area:
            best_overall_min_area = best_run_min_area
            best_overall_points = best_run_points

    return best_overall_points, best_overall_min_area


# EVOLVE-BLOCK-END


# This part remains fixed (not evolved)
def run_search_point(n=13):
    """
    Run a search for a given number of points.
    This is the fixed entry point for the evaluator.
    """
    points, min_area = find_best_placement(n)
    return points, min_area


if __name__ == "__main__":
    """Run the Heilbronn problem for convex regions for n=13"""
    n = 13  # Number of points
    optimized_points, min_area = run_search_point(n)

    print(f"Optimized minimum triangle area for {n} points: {min_area:.8f}")
    print("Optimized points configuration:")
    for i, p in enumerate(optimized_points):
        print(f"  P{i + 1}: ({p[0]:.8f}, {p[1]:.8f})")
