"""The Heilbronn problem for convex regions"""
import math
import random
import time
from scipy.spatial import ConvexHull
import numpy as np  # Used by ConvexHull, explicitly importing for clarity

# Global constants for precision and region definition
TOL = 1e-6  # Tolerance for checking if points are too close
TOL_SQ = TOL * TOL  # Squared tolerance for distance comparisons
REGION_AREA = 1.0  # Unit square area (assuming a unit square for the region)

# New constant for minimum valid triangle area.
# This value must match the check in verify_solution to ensure consistency.
MIN_VALID_TRIANGLE_AREA = 1e-10


def points_are_close(p1, p2):
    """Check if two points are closer than a tolerance threshold."""
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    return dx * dx + dy * dy < TOL_SQ


def triangle_area(a, b, c):
    """Calculate area of triangle given three vertices using cross product."""
    # Using abs to ensure positive area, as orientation doesn't matter for area
    return 0.5 * abs((b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1]))


def random_initial_placement(n):
    """Random initialization with collision avoidance."""
    points = []
    # Using TOL_SQ for initial point separation, matching the general tolerance.
    # This allows points to be closer initially, giving SA more flexibility in starting configurations.
    initial_min_dist_sq = TOL_SQ

    while len(points) < n:
        x = random.uniform(0, 1)
        y = random.uniform(0, 1)
        new_point = (x, y)
        too_close = False
        for p in points:
            dx = new_point[0] - p[0]
            dy = new_point[1] - p[1]
            if dx * dx + dy * dy < initial_min_dist_sq:
                too_close = True
                break
        if not too_close:
            points.append(new_point)
    return points


def initial_placement(n):
    """Initializes points, ensuring no degenerate triangles in the initial configuration."""
    max_retries = (
        1000  # Limit retries to prevent infinite loops, though unlikely for n=13
    )
    for _ in range(max_retries):
        points = random_initial_placement(
            n
        )  # Always use random placement for robustness

        if n < 3:  # If less than 3 points, no triangles can be formed, so it's "valid"
            return points

        # Check if the initial configuration has any degenerate triangles
        has_degenerate = False
        # Only iterate if there are at least 3 points to form a triangle
        if n >= 3:
            # This is a brute-force check, but critical for a valid starting point
            for i in range(n):
                for j in range(i + 1, n):
                    for k in range(j + 1, n):
                        area = triangle_area(points[i], points[j], points[k])
                        if area < MIN_VALID_TRIANGLE_AREA:
                            has_degenerate = True
                            break  # Found a degenerate triangle, retry
                    if has_degenerate:
                        break
                if has_degenerate:
                    break

        if not has_degenerate:
            return points

    # Fallback if too many retries fail (should be extremely rare for n=13)
    print(
        f"Warning: Failed to find non-degenerate initial placement after {max_retries} retries. Proceeding with potentially degenerate points."
    )
    return random_initial_placement(n)  # Return a random set anyway as a last resort


def precompute_triangles(points):
    """Precompute all triangles and point-triangle mappings."""
    n = len(points)
    triangles = []  # List of (i, j, k) triplets representing point indices
    areas = []  # Corresponding triangle areas
    point_triangles = [[] for _ in range(n)]  # Triangles per point

    # Iterate through all unique combinations of three points to form triangles
    tri_idx = 0
    if n >= 3:  # Only create triangles if there are at least 3 points
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    triangles.append((i, j, k))
                    # Store which triangles involve each point for efficient updates
                    point_triangles[i].append(tri_idx)
                    point_triangles[j].append(tri_idx)
                    point_triangles[k].append(tri_idx)
                    areas.append(triangle_area(points[i], points[j], points[k]))
                    tri_idx += 1

    return triangles, areas, point_triangles


def simulate_heilbronn(n=13, num_runs=15):
    """Optimized Heilbronn solver using simulated annealing with adaptive mechanisms."""
    best_overall_min_area = 0.0
    best_overall_points = []

    # Tuned parameters for Simulated Annealing
    MAX_ITER_PER_RUN = 750_000

    # T0_INIT Adjusted: From 0.08 to 0.1 for a hotter start, allowing broader exploration initially.
    T0_INIT = 0.1

    # DECAY_RATE Adjusted: From 0.999985 to 0.99998 for a slightly faster cooling schedule.
    # This aims for quicker convergence to a good solution after the initial broad search.
    DECAY_RATE = 0.99998

    for run in range(num_runs):
        points = initial_placement(
            n
        )  # Initialize points for the current run, ensuring non-degeneracy
        triangles, areas, point_triangles = precompute_triangles(
            points
        )  # Precompute triangle data

        # Calculate initial minimum area. initial_placement guarantees it's >= MIN_VALID_TRIANGLE_AREA
        current_min_area = min(areas) if areas else 0.0

        # Initialize best state for the current run
        best_run_points = list(points)
        best_run_min_area = current_min_area

        current_energy = (
            -current_min_area
        )  # Energy is negative of min_area (maximize min_area -> minimize energy)
        T = T0_INIT  # Current temperature
        max_step = 0.05  # Current maximum step size for point movement
        accepted_count = 0  # Counter for accepted moves for adaptive step size
        total_move_attempts = (
            0  # Counter for total valid move attempts for adaptive step size
        )

        for iter_count in range(MAX_ITER_PER_RUN):
            # --- Point Selection Strategy: Targeted Move vs. Random Move ---
            # In early stages or if current_min_area is tiny, allow more random exploration.
            # Otherwise, focus on moving points that are part of the smallest triangles.
            if current_min_area < 1e-9 or iter_count < MAX_ITER_PER_RUN / 10:
                idx = random.randint(0, n - 1)
            else:
                # Find all triangles that achieve the current minimum area (within a small tolerance)
                min_tri_indices = [
                    i
                    for i, area_val in enumerate(areas)
                    if abs(area_val - current_min_area) < 1e-8
                ]

                # Collect all unique points involved in these minimum triangles
                min_area_points_candidates = set()
                for tri_idx_val in min_tri_indices:
                    i, j, k = triangles[tri_idx_val]
                    min_area_points_candidates.add(i)
                    min_area_points_candidates.add(j)
                    min_area_points_candidates.add(k)

                if min_area_points_candidates:
                    idx = random.choice(list(min_area_points_candidates))
                else:  # Fallback if no specific min-area triangles found (shouldn't happen with valid points)
                    idx = random.randint(0, n - 1)

            old_point = points[idx]

            # Determine step size for the current move attempt
            current_max_step_for_move = max_step

            # Generate new potential position for the point
            dx = random.uniform(-current_max_step_for_move, current_max_step_for_move)
            dy = random.uniform(-current_max_step_for_move, current_max_step_for_move)
            new_x = max(0.0, min(1.0, old_point[0] + dx))  # Clamp coordinates to [0, 1]
            new_y = max(0.0, min(1.0, old_point[1] + dy))
            new_point = (new_x, new_y)

            # Check for collision with other points
            too_close = False
            for i, p in enumerate(points):
                if i == idx:
                    continue  # Skip checking against itself
                if points_are_close(new_point, p):
                    too_close = True
                    break

            if too_close:
                continue  # Skip to next iteration without attempting move

            # Backup affected triangle areas before modifying the point
            backup_areas = [areas[tri_idx] for tri_idx in point_triangles[idx]]
            old_min_area_before_move = (
                current_min_area  # Store global min before this move attempt
            )

            # Tentatively update the point and recompute areas of affected triangles
            points[idx] = new_point
            min_changed_area = float(
                "inf"
            )  # Minimum area among triangles involving the moved point
            for tri_idx in point_triangles[idx]:
                i, j, k = triangles[tri_idx]
                area_val = triangle_area(points[i], points[j], points[k])
                areas[tri_idx] = area_val
                if area_val < min_changed_area:
                    min_changed_area = area_val

            # Reject the move if it creates any degenerate triangles (area near zero).
            if min_changed_area < MIN_VALID_TRIANGLE_AREA:
                # Revert point and areas to their state before the move
                points[idx] = old_point
                for tri_idx, area_val in zip(point_triangles[idx], backup_areas):
                    areas[tri_idx] = area_val
                continue  # Skip to next iteration

            # Recompute global minimum area after the move.
            current_min_area = min(areas)
            new_energy = -current_min_area  # New energy based on new min_area
            delta_energy = new_energy - current_energy  # Change in energy

            # Simulated Annealing acceptance criteria
            total_move_attempts += 1
            if delta_energy < 0 or random.random() < math.exp(-delta_energy / T):
                # Move accepted (either improved energy or accepted stochastically)
                accepted_count += 1
                current_energy = new_energy
                if current_min_area > best_run_min_area:
                    # Found a new best minimum area for this run
                    best_run_min_area = current_min_area
                    best_run_points = list(points)  # Update best points
            else:
                # Move rejected, revert point and areas
                points[idx] = old_point
                for tri_idx, area_val in zip(point_triangles[idx], backup_areas):
                    areas[tri_idx] = area_val
                current_min_area = old_min_area_before_move  # Restore global min area

            # Decay temperature after every iteration
            T = T0_INIT * (DECAY_RATE**iter_count)

            # Adaptive step size adjustment based on recent acceptance rate
            if iter_count % 50 == 0 and total_move_attempts > 0:
                accept_rate = accepted_count / total_move_attempts
                if accept_rate < 0.2:
                    max_step *= 0.85
                elif accept_rate > 0.6:
                    max_step *= 1.15
                # max_step upper bound Adjusted: From 0.15 to 0.2 to allow for larger, more explorative steps.
                max_step = max(0.001, min(max_step, 0.2))
                accepted_count = 0
                total_move_attempts = 0

            # Early exit if temperature becomes extremely low (system has essentially frozen)
            if T < 1e-10:
                break

        # After each run, compare the best result with the overall best found across all runs
        if best_run_min_area > best_overall_min_area:
            best_overall_min_area = best_run_min_area
            best_overall_points = list(best_run_points)

    # Calculate the best_area_ratio using the overall best points
    if len(best_overall_points) >= 3:
        try:
            # If the minimum area found was very small, ConvexHull might still fail or yield a small area.
            # Only compute if the min_area is considered valid.
            if best_overall_min_area < MIN_VALID_TRIANGLE_AREA:
                convex_hull_area = (
                    0.0  # If min_area is degenerate, so is the hull for ratio purpose
                )
            else:
                # For 2D points, ConvexHull.volume attribute gives the area of the convex hull.
                convex_hull_area = ConvexHull(np.array(best_overall_points)).volume

        except (
            Exception
        ):  # Handle potential errors like all points collinear or too few points
            convex_hull_area = 0.0

        if convex_hull_area > TOL:  # Avoid division by zero or very small hull areas
            best_area_ratio = best_overall_min_area / convex_hull_area
        else:
            best_area_ratio = 0.0
    else:
        best_area_ratio = 0.0

    return best_overall_points, best_overall_min_area, best_area_ratio


def verify_solution(points, min_area):
    """Validate solution correctness and constraints."""
    n = len(points)

    # 1. Check all points are within [0,1]x[0,1] (with a small tolerance)
    for p in points:
        if not (-TOL <= p[0] <= 1 + TOL and -TOL <= p[1] <= 1 + TOL):
            print(f"Verification FAILED: Point {p} out of bounds.")
            return False

    # 2. Check minimum point separation
    for i in range(n):
        for j in range(i + 1, n):
            if points_are_close(points[i], points[j]):
                print(
                    f"Verification FAILED: Points {points[i]} and {points[j]} are too close."
                )
                return False

    # 3. Verify actual minimum area matches reported
    min_computed = float("inf")
    if n < 3:  # If less than 3 points, no triangles can be formed
        return min_area == 0.0  # Expected min_area is 0.0 in this case

    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                area = triangle_area(points[i], points[j], points[k])
                # This is the strict check for degenerate triangles.
                if area < MIN_VALID_TRIANGLE_AREA:
                    print(
                        f"Verification FAILED: Degenerate triangle found with points {points[i]}, {points[j]}, {points[k]} (area: {area:.10f})."
                    )
                    return False
                if area < min_computed:
                    min_computed = area

    # Check if the computed minimum area is close to the reported minimum area
    if (
        abs(min_computed - min_area) < 5e-6
    ):  # Tolerance for reported vs. computed min_area
        return True
    else:
        print(
            f"Verification FAILED: Reported min area ({min_area:.10f}) does not match computed min area ({min_computed:.10f})."
        )
        return False


# Example usage
if __name__ == "__main__":
    n = 13
    start_time = time.time()
    # Call simulate_heilbronn with updated num_runs
    points, min_area, best_area_ratio = simulate_heilbronn(n, num_runs=15)
    end_time = time.time()

    print(f"Simulation finished in {end_time - start_time:.2f} seconds.")
    print(f"Optimized minimum triangle area: {min_area:.6f}")
    print(f"Best area ratio (min_area / convex_hull_area): {best_area_ratio:.6f}")
    print(
        f"Verification: {'PASSED' if verify_solution(points, min_area) else 'FAILED'}"
    )
    # Optional: Print the optimized point coordinates
    for i, p in enumerate(points):
        print(f"Point {i+1}: ({p[0]:.8f}, {p[1]:.8f})")
