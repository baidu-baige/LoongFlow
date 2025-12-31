"""Best"""

import numpy as np
import scipy as sp
from scipy.optimize import basinhopping
from scipy.spatial import Voronoi


def optimize_construct(n=16, d=2):
    """
    Find n points in d-dimensional Euclidean space that minimize the ratio R = D_max / D_min,
    using circle-packed initialization and Voronoi-enhanced basinhopping.
    """

    # 1. Corrected Circle Packing Initialization
    def circle_packing_initialization(n, d):
        if d == 2 and n == 16:
            # Proper hexagonal packing for 16 points
            points = []
            hex_spacing = 1.0
            sqrt3 = np.sqrt(3)

            # Create 4 rows with 4 points each
            for row in range(4):
                for col in range(4):
                    x = col * 2 * hex_spacing
                    y = row * sqrt3 * hex_spacing

                    # Offset every other row
                    if row % 2 == 1:
                        x += hex_spacing
                    points.append([x, y])

            points = np.array(points, dtype=np.float64)

            # Center and normalize
            centroid = np.mean(points, axis=0)
            points -= centroid
            dists = sp.spatial.distance.pdist(points)
            min_dist = np.min(dists)
            return points / min_dist
        else:
            # Fallback to grid initialization
            grid_size = int(np.ceil(np.sqrt(n)))
            points = []
            for i in range(grid_size):
                for j in range(grid_size):
                    if len(points) < n:
                        points.append([float(i), float(j)])
            points = np.array(points, dtype=np.float64)
            centroid = np.mean(points, axis=0)
            points -= centroid
            dists = sp.spatial.distance.pdist(points)
            min_dist = np.min(dists)
            return points / min_dist

    points = circle_packing_initialization(n, d)

    # 2. Tuned Physics Pre-optimization
    dt = 0.05
    damping = 0.95  # Increased damping
    velocities = np.zeros_like(points, dtype=np.float64)
    L = 2.5 * np.sqrt(n)  # Smaller boundary

    for iter in range(50):
        # Vectorized distance calculation
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=-1))
        np.fill_diagonal(dists, np.inf)

        # Softer force calculation (1/r)
        with np.errstate(divide="ignore", invalid="ignore"):
            force_mags = np.where(dists > 1e-8, 1.0 / dists, 0)
        force_vecs = diff * force_mags[..., np.newaxis]

        # Sum forces and update
        forces = np.sum(force_vecs, axis=1)

        # Gentler boundary force
        boundary_force = -0.05 * np.sign(points) * np.minimum(np.abs(points), L)
        forces += boundary_force

        # Update with gradient clipping
        velocities = velocities + forces * dt
        velocity_norms = np.linalg.norm(velocities, axis=1)
        max_vel = 0.3  # Reduced max velocity
        clip_mask = velocity_norms > max_vel
        velocities[clip_mask] = (
            velocities[clip_mask].T / velocity_norms[clip_mask] * max_vel
        ).T

        points += velocities * dt

        # Apply damping every 5 steps
        if iter % 5 == 0:
            velocities *= damping

    # 3. Improved Voronoi-enhanced Basinhopping
    def objective(x):
        points_in = x.reshape(n, d)
        dists = sp.spatial.distance.pdist(points_in)
        if len(dists) == 0:
            return 1e20
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_min < 1e-8:
            return 1e20
        return (d_max / d_min) ** 2

    # Relaxed constraint
    def min_distance_constraint(x):
        points_in = x.reshape(n, d)
        dists = sp.spatial.distance.pdist(points_in)
        if len(dists) == 0:
            return -1e10
        return np.min(dists) - 0.5  # Less strict constraint

    # More conservative Voronoi mutations
    def take_step(x):
        points_current = x.reshape(n, d).copy()

        if np.random.rand() < 0.2:  # Reduced Voronoi mutation probability
            i = np.random.randint(n)
            try:
                vor = Voronoi(points_current)
                region_index = vor.point_region[i]
                region = vor.regions[region_index]
                if region and -1 not in region:
                    vertices = vor.vertices[region]
                    if len(vertices) > 0:
                        # Move only 25% toward vertex
                        new_pos = (
                            0.75 * points_current[i]
                            + 0.25 * vertices[np.random.choice(len(vertices))]
                        )
                        points_current[i] = new_pos
            except:
                pass
        else:
            # Smaller Gaussian step
            points_current += np.random.normal(0, 0.05, (n, d))

        return points_current.flatten()

    # Run basinhopping with improved parameters
    res = basinhopping(
        objective,
        points.flatten(),
        minimizer_kwargs={
            "method": "SLSQP",
            "constraints": [{"type": "ineq", "fun": min_distance_constraint}],
            "options": {"maxiter": 500, "ftol": 1e-8},
        },
        niter=70,  # More iterations
        T=0.3,  # Lower temperature
        stepsize=0.05,  # Smaller stepsize
        take_step=take_step,
        callback=lambda x, f, accept: f < 12.89,  # Early termination
    )

    # Extract and normalize results
    points_refined = res.x.reshape(n, d)
    dists_refined = sp.spatial.distance.pdist(points_refined)
    min_dist_refined = np.min(dists_refined)
    max_dist_refined = np.max(dists_refined)
    ratio_squared = (max_dist_refined / min_dist_refined) ** 2

    # Final normalization to set D_min = 1
    return points_refined / min_dist_refined, ratio_squared


def max_min_distance_ratio(points: np.ndarray):
    """Calculate the ratio of max distance to min distance"""
    pairwise_distances = sp.spatial.distance.pdist(points)

    if len(pairwise_distances) == 0:
        return 0.0

    min_distance = np.min(pairwise_distances)
    max_distance = np.max(pairwise_distances)

    if abs(min_distance) < 1e-10 or abs(max_distance) < 1e-10:
        return 0.0

    ratio_squared = (max_distance / min_distance) ** 2
    if ratio_squared < 1e-10:
        return 0.0

    return ratio_squared


if __name__ == "__main__":
    """Main function"""

    def verification1(points: np.ndarray):
        """Verify the correctness of the generated points"""
        pairwise_distances = sp.spatial.distance.pdist(points)
        # Handle the case where there are no valid distances (e.g., all points are identical)
        if pairwise_distances.size == 0:
            return False, None

        min_distance = np.min(pairwise_distances)
        max_distance = np.max(pairwise_distances)
        if abs(min_distance) < 1e-10 or abs(max_distance) < 1e-10:
            return False, None

        ratio_squared = (max_distance / min_distance) ** 2
        if ratio_squared is None or ratio_squared < 1e-10:
            return False, None

        return True, ratio_squared

    best = None
    best_points = None
    for i in range(2000):
        print(f"Iteration: {i}")
        points, ratio_squared = optimize_construct(16, 2)
        valid, final_ratio = verification1(points)
        if not valid:
            print("Invalid points found. Continuing with next iteration.")
            continue
        print(f"Construction has {len(points)} points in {points.shape[1]} dimensions.")
        print(points)
        print(f"Ratio squared: {final_ratio}")
        if best is None or final_ratio < best:
            best = final_ratio
            best_points = points
            print(f"New best ratio found:{best}")
    print(f"Final Ratio Squared: {best}")
    if best_points is not None:
        points_list = [tuple(map(float, pt)) for pt in best_points]
        print("Best points:")
        print(points_list)
    else:
        print("No valid points found.")
