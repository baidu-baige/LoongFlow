"""Heilbronn triangle problem"""
import numpy as np
import math
import itertools
from scipy.optimize import minimize
import time


def run_search_point(n=11):
    """
    Solves the Heilbronn triangle problem for n points in an equilateral triangle.
    The goal is to maximize the minimum area of any triangle formed by three points.
    Target benchmark for n=11 is a normalized min_area of 0.0365.
    """

    def find_best_placement(n):
        # Geometric constants for the equilateral triangle with vertices (0,0), (1,0), (0.5, sqrt(3)/2)
        SQRT3 = 1.7320508075688772
        TRI_AREA = SQRT3 / 4.0
        # Normalization factor: Normalized Area = Area / (SQRT3 / 4) = 2 * Area / (SQRT3 / 2)
        NORM = SQRT3 / 2.0

        # Pre-calculate all triplet combinations for vectorized area computation
        triplet_indices = np.array(list(itertools.combinations(range(n), 3)))
        ti = triplet_indices[:, 0]
        tj = triplet_indices[:, 1]
        tk = triplet_indices[:, 2]

        def get_sa2_vectorized(pts):
            """Returns the signed double area (2*Area) for all 165 triplets."""
            # Determinant formula: x1(y2-y3) + x2(y3-y1) + x3(y1-y2)
            return (
                pts[ti, 0] * (pts[tj, 1] - pts[tk, 1])
                + pts[tj, 0] * (pts[tk, 1] - pts[ti, 1])
                + pts[tk, 0] * (pts[ti, 1] - pts[tj, 1])
            )

        def project(pts):
            """Ensures all points are strictly within the equilateral triangle boundaries."""
            p = pts.copy()
            eps = 1e-12
            # Boundary y is between 0 and sqrt(3)/2
            p[:, 1] = np.clip(p[:, 1], eps, (SQRT3 / 2.0) - eps)
            for i in range(p.shape[0]):
                # x bound depends on y: [y/sqrt(3), 1 - y/sqrt(3)]
                limit = p[i, 1] / SQRT3
                p[i, 0] = np.clip(p[i, 0], limit + eps, 1.0 - limit - eps)
            return p

        # Specialized seeding strategies to explore diverse basins
        def get_random_seed():
            u1, u2 = np.random.rand(n), np.random.rand(n)
            s1 = np.sqrt(u1)
            pts = np.zeros((n, 2))
            pts[:, 0] = (1 - s1) * 0 + (s1 * (1 - u2)) * 1 + (s1 * u2) * 0.5
            pts[:, 1] = (1 - s1) * 0 + (s1 * (1 - u2)) * 0 + (s1 * u2) * (SQRT3 / 2)
            return project(pts)

        def get_d1_seed():
            """Reflection symmetry (x -> 1-x)."""
            half = (n - 1) // 2
            u1, u2 = np.random.rand(half), np.random.rand(half)
            s1 = np.sqrt(u1)
            pts_half = np.zeros((half, 2))
            pts_half[:, 0] = (1 - s1) * 0 + (s1 * (1 - u2)) * 1 + (s1 * u2) * 0.5
            pts_half[:, 1] = (
                (1 - s1) * 0 + (s1 * (1 - u2)) * 0 + (s1 * u2) * (SQRT3 / 2)
            )
            pts = np.zeros((n, 2))
            for i in range(half):
                pts[i] = pts_half[i]
                pts[half + i] = [1.0 - pts_half[i, 0], pts_half[i, 1]]
            pts[n - 1] = [0.5, np.random.rand() * (SQRT3 / 2.0)]
            return project(pts)

        def get_c3_seed():
            """Rotational symmetry (120 degrees)."""
            centroid = np.array([0.5, SQRT3 / 6.0])
            pts = np.zeros((n, 2))
            pts[0] = centroid
            angle = 2 * math.pi / 3
            rot = np.array(
                [
                    [math.cos(angle), -math.sin(angle)],
                    [math.sin(angle), math.cos(angle)],
                ]
            )
            for i in range(3):
                u1, u2 = np.random.rand(), np.random.rand()
                s1 = math.sqrt(u1)
                p = (
                    (1 - s1) * np.array([0, 0])
                    + (s1 * (1 - u2)) * np.array([1, 0])
                    + (s1 * u2) * np.array([0.5, SQRT3 / 2])
                )
                p_rel = p - centroid
                pts[1 + 3 * i] = p
                pts[2 + 3 * i] = (rot @ p_rel) + centroid
                pts[3 + 3 * i] = (rot @ (rot @ p_rel)) + centroid
            pts[10] = [0.5, np.random.rand() * (SQRT3 / 2.0)]
            return project(pts)

        def optimize(init_pts, iters=100, ftol=1e-7, active_set=None):
            """Performs SLSQP optimization with a slack variable to maximize min area."""
            curr_pts = project(init_pts)
            for pass_idx in range(2):  # Double pass to handle sign/orientation flips
                sa2 = get_sa2_vectorized(curr_pts)
                signs = np.sign(sa2)
                signs[signs == 0] = 1.0

                if active_set is not None:
                    # Focus on the smallest triangles to speed up computation
                    indices = np.argsort(np.abs(sa2))[:active_set]
                    a_ti, a_tj, a_tk = ti[indices], tj[indices], tk[indices]
                    a_signs = signs[indices]
                else:
                    a_ti, a_tj, a_tk = ti, tj, tk
                    a_signs = signs

                def objective(X):
                    return -X[-1]

                def constraints(X):
                    pts_flat = X[:-1].reshape((n, 2))
                    t = X[-1]
                    px, py = pts_flat[:, 0], pts_flat[:, 1]
                    # Boundary constraints: y >= 0, sqrt(3)x - y >= 0, sqrt(3)(1-x) - y >= 0
                    c_bound = np.concatenate(
                        [py, SQRT3 * px - py, SQRT3 * (1.0 - px) - py]
                    )
                    # Area constraints: (sign * 2 * Area / Normalization) >= t
                    sa2_active = (
                        pts_flat[a_ti, 0] * (pts_flat[a_tj, 1] - pts_flat[a_tk, 1])
                        + pts_flat[a_tj, 0] * (pts_flat[a_tk, 1] - pts_flat[a_ti, 1])
                        + pts_flat[a_tk, 0] * (pts_flat[a_ti, 1] - pts_flat[a_tj, 1])
                    )
                    c_area = (a_signs * sa2_active / NORM) - t
                    return np.concatenate([c_bound, c_area])

                x0 = np.concatenate([curr_pts.flatten(), [np.min(np.abs(sa2)) / NORM]])
                res = minimize(
                    objective,
                    x0,
                    method="SLSQP",
                    constraints={"type": "ineq", "fun": constraints},
                    options={"maxiter": iters, "ftol": ftol, "disp": False},
                )

                # Check for improvement even if not perfectly convergent
                cand_pts = res.x[:-1].reshape((n, 2))
                cand_val = np.min(np.abs(get_sa2_vectorized(cand_pts))) / NORM
                if cand_val >= (np.min(np.abs(sa2)) / NORM):
                    curr_pts = cand_pts
                else:
                    break

            final_sa2 = get_sa2_vectorized(curr_pts)
            return curr_pts, np.min(np.abs(final_sa2)) / NORM

        # Variables for global search tracking
        best_pts = None
        best_val = -1.0
        start_time = time.time()
        np.random.seed(int(time.time() * 1000) % 2**32)

        # Main optimization loop: Iterate until time limit is reached
        while time.time() - start_time < 53:
            # Seed selection logic
            r = np.random.rand()
            if r < 0.2:
                seed = get_random_seed()
            elif r < 0.45:
                seed = get_d1_seed()
            elif r < 0.55:
                seed = get_c3_seed()
            else:
                if best_pts is not None:
                    # Explore neighborhood of current best
                    seed = project(best_pts + np.random.normal(0, 0.012, (n, 2)))
                else:
                    seed = get_random_seed()

            # Phase 1: Rapid exploration using active-set constraint selection
            pts, val = optimize(seed, iters=80, ftol=1e-4, active_set=70)

            # Phase 2: Full refinement for promising candidates
            if val > best_val:
                pts, val = optimize(pts, iters=150, ftol=1e-9)
                if val > best_val:
                    best_val = val
                    best_pts = pts

        # Phase 3: High-precision polish for the absolute best result
        if best_pts is not None:
            best_pts, best_val = optimize(best_pts, iters=400, ftol=1e-12)

        return project(best_pts), best_val

    # Execute search
    points, min_area = find_best_placement(n)
    return points, min_area


if __name__ == "__main__":
    n = 11
    pts, ma = run_search_point(n)
    print(f"Final Min Area (Normalized): {ma:.8f}")
