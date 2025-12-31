import sys
import time
import numpy as np
from scipy.optimize import minimize, linprog


class PackingOptimizer:
    """
    Optimizer for packing circles in a unit square to maximize total area.
    
    This class implements constrained optimization techniques to find optimal
    circle packing configurations. It uses:
    - Nonlinear optimization with analytical gradients
    - Linear programming for radius maximization
    - Basin hopping for global optimization
    
    Attributes:
        n (int): Number of circles to pack
        triu_idx (tuple): Indices for upper triangle of distance matrix
        n_pairs (int): Number of unique circle pairs
    """
    def __init__(self, n_circles):
        """
        Initialize the packing optimizer.
        
        Args:
            n_circles (int): Number of circles to pack in the unit square
        """
        self.n = n_circles
        # Indices for upper triangle of distance matrix (i < j)
        self.triu_idx = np.triu_indices(n_circles, k=1)
        self.n_pairs = len(self.triu_idx[0])

    def get_initial_guess(self, mode="random"):
        """
        Generate initial circle positions and radii.
        
        Args:
            mode (str): Initialization mode - "random" or "ring"
            
        Returns:
            tuple: (centers, radii) where:
                - centers: (n,2) array of (x,y) coordinates
                - radii: (n,) array of circle radii
        """
        n = self.n
        if mode == "ring":
            # Construct a structured heuristic (concentric rings)
            centers = np.zeros((n, 2))
            centers[0] = [0.5, 0.5]
            cnt = 1

            # Inner ring
            if n > 1:
                m = min(8, n - 1)
                th = np.linspace(0, 2 * np.pi, m, endpoint=False)
                centers[cnt : cnt + m, 0] = 0.5 + 0.2 * np.cos(th)
                centers[cnt : cnt + m, 1] = 0.5 + 0.2 * np.sin(th)
                cnt += m

            # Outer ring
            if n > cnt:
                m = min(16, n - cnt)
                th = np.linspace(0, 2 * np.pi, m, endpoint=False)
                centers[cnt : cnt + m, 0] = 0.5 + 0.4 * np.cos(th)
                centers[cnt : cnt + m, 1] = 0.5 + 0.4 * np.sin(th)
                cnt += m

            # Remaining random
            if n > cnt:
                centers[cnt:] = np.random.rand(n - cnt, 2)

            radii = np.full(n, 0.05)
            return centers, radii
        else:
            # Random initialization
            return np.random.rand(n, 2), np.full(n, 0.01)

    def obj_func(self, x):
        """
        Objective function: minimize negative sum of radii (maximize total area).
        
        Args:
            x: Parameter vector containing [x-coords, y-coords, radii]
            
        Returns:
            float: Negative sum of all circle radii
        """
        # Minimize negative sum of radii (equivalent to maximizing total area)
        return -np.sum(x[2 * self.n :])

    def obj_jac(self, x):
        """
        Gradient of the objective function.
        
        Args:
            x: Parameter vector containing [x-coords, y-coords, radii]
            
        Returns:
            array: Gradient vector where:
                - Positions for radii have gradient -1.0
                - All other elements have gradient 0.0
        """
        # Gradient of objective: derivative w.r.t. each radius is -1
        grad = np.zeros_like(x)
        grad[2 * self.n :] = -1.0
        return grad

    def constraints(self, x):
        """
        Construct all constraint functions for the optimization problem.
        
        Constraints include:
        - Boundary constraints (left, right, bottom, top walls)
        - Positive radius constraints
        - Non-overlapping circle constraints
        
        Args:
            x: Parameter vector containing [x-coords, y-coords, radii]
            
        Returns:
            array: Combined constraint violations (must be >= 0 for feasibility)
        """
        n = self.n
        xc = x[:n]
        yc = x[n : 2 * n]
        r = x[2 * n :]

        # 1. Boundary constraints (5*n)
        # x >= r, 1-x >= r, y >= r, 1-y >= r, r >= 0
        b_cons = np.concatenate(
            [
                xc - r,  # Left
                1.0 - xc - r,  # Right
                yc - r,  # Bottom
                1.0 - yc - r,  # Top
                r,  # Positive radius
            ]
        )

        # 2. Overlap constraints (n*(n-1)/2)
        # dist^2 >= (ri + rj)^2
        dx = xc[self.triu_idx[0]] - xc[self.triu_idx[1]]
        dy = yc[self.triu_idx[0]] - yc[self.triu_idx[1]]
        d2 = dx**2 + dy**2
        r_sum = r[self.triu_idx[0]] + r[self.triu_idx[1]]
        ov_cons = d2 - r_sum**2

        return np.concatenate([b_cons, ov_cons])

    def constraints_jac(self, x):
        """
        Jacobian matrix of the constraint functions.
        
        Computes the gradient of all constraints with respect to each variable.
        
        Args:
            x: Parameter vector containing [x-coords, y-coords, radii]
            
        Returns:
            array: Jacobian matrix of shape (num_constraints, 3*n)
        """
        n = self.n
        xc = x[:n]
        yc = x[n : 2 * n]
        r = x[2 * n :]

        total_cons = 5 * n + self.n_pairs
        jac = np.zeros((total_cons, 3 * n))
        eye = np.eye(n)

        # --- Boundary Jacobians ---
        # 0: x - r
        jac[0:n, 0:n] = eye
        jac[0:n, 2 * n :] = -eye

        # 1: 1 - x - r
        jac[n : 2 * n, 0:n] = -eye
        jac[n : 2 * n, 2 * n :] = -eye

        # 2: y - r
        jac[2 * n : 3 * n, n : 2 * n] = eye
        jac[2 * n : 3 * n, 2 * n :] = -eye

        # 3: 1 - y - r
        jac[3 * n : 4 * n, n : 2 * n] = -eye
        jac[3 * n : 4 * n, 2 * n :] = -eye

        # 4: r >= 0
        jac[4 * n : 5 * n, 2 * n :] = eye

        # --- Overlap Jacobians ---
        start = 5 * n
        rows = np.arange(start, start + self.n_pairs)
        idx_i = self.triu_idx[0]
        idx_j = self.triu_idx[1]

        dx = xc[idx_i] - xc[idx_j]
        dy = yc[idx_i] - yc[idx_j]
        r_sum = r[idx_i] + r[idx_j]

        # Partial wrt x_i: 2*(xi - xj)
        jac[rows, idx_i] = 2 * dx
        jac[rows, idx_j] = -2 * dx

        # Partial wrt y_i: 2*(yi - yj)
        jac[rows, n + idx_i] = 2 * dy
        jac[rows, n + idx_j] = -2 * dy

        # Partial wrt r_i: -2*(ri + rj)
        term = -2 * r_sum
        jac[rows, 2 * n + idx_i] = term
        jac[rows, 2 * n + idx_j] = term

        return jac

    def polish_radii(self, centers):
        """
        Use Linear Programming to maximize radii for fixed centers.
        
        This method finds the maximum possible radii for given centers while
        satisfying all constraints (boundary and non-overlapping).
        
        Args:
            centers: (n,2) array of fixed circle centers
            
        Returns:
            tuple: (radii, total_radius) where:
                - radii: Optimized radii array
                - total_radius: Sum of all radii
        """
        n = self.n
        # Maximize sum(r) <=> Minimize -sum(r)
        c = -np.ones(n)

        # Constraints: r_i + r_j <= Dist_ij - epsilon
        # Use a larger epsilon to account for solver tolerances and check strictness
        epsilon = 1e-6

        A_ub = np.zeros((self.n_pairs, n))
        idx_i = self.triu_idx[0]
        idx_j = self.triu_idx[1]

        # Row k: r[i] + r[j] <= dist
        row_indices = np.arange(self.n_pairs)
        A_ub[row_indices, idx_i] = 1.0
        A_ub[row_indices, idx_j] = 1.0

        dists = np.sqrt(np.sum((centers[idx_i] - centers[idx_j]) ** 2, axis=1))
        b_ub = dists - epsilon

        # Box Bounds: 0 <= r_i <= min(wall_dist)
        # wall_dist = min(x, 1-x, y, 1-y)
        min_x = np.minimum(centers[:, 0], 1 - centers[:, 0])
        min_y = np.minimum(centers[:, 1], 1 - centers[:, 1])
        max_r_wall = np.minimum(min_x, min_y)
        # Ensure non-negative
        max_r_wall = np.maximum(0, max_r_wall)

        # linprog bounds
        bounds = [(0, u) for u in max_r_wall]

        try:
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
            if res.success:
                return res.x, -res.fun
        except Exception:
            pass

        # Fallback
        return np.zeros(n), 0.0

    def cleanup_solution(self, centers, radii):
        """
        Final safety net to ensure strict feasibility.
        
        Iteratively shrinks radii to guarantee:
        - All circles stay within unit square
        - No overlapping circles
        - Radii remain non-negative
        
        Args:
            centers: (n,2) array of circle centers
            radii: (n,) array of circle radii
            
        Returns:
            array: Strictly feasible radii
        """
        xc = centers[:, 0]
        yc = centers[:, 1]
        rd = radii.copy()

        # 1. Enforce Boundaries
        rd = np.minimum(rd, xc)
        rd = np.minimum(rd, 1 - xc)
        rd = np.minimum(rd, yc)
        rd = np.minimum(rd, 1 - yc)
        rd = np.maximum(rd, 0)

        # 2. Enforce Overlaps strictly
        epsilon = 1e-12
        for _ in range(100):
            dirty = False
            for k in range(self.n_pairs):
                i = self.triu_idx[0][k]
                j = self.triu_idx[1][k]

                dist = np.sqrt((xc[i] - xc[j]) ** 2 + (yc[i] - yc[j]) ** 2)
                sum_r = rd[i] + rd[j]

                if sum_r > dist:  # Strict check
                    dirty = True
                    # Shrink proportionally
                    target = max(0, dist - epsilon)
                    if sum_r > 1e-15:
                        scale = target / sum_r
                        rd[i] *= scale
                        rd[j] *= scale
                    else:
                        rd[i] = 0
                        rd[j] = 0
            if not dirty:
                break
        return rd

    def local_minimize(self, centers, radii, maxiter=50):
        """
        Perform local optimization using SLSQP algorithm.
        
        This method optimizes circle positions and radii subject to constraints,
        then polishes the solution using linear programming.
        
        Args:
            centers: (n,2) array of initial circle centers
            radii: (n,) array of initial circle radii
            maxiter: Maximum number of iterations for SLSQP
            
        Returns:
            tuple: (optimized_centers, optimized_radii, total_radius)
        """
        n = self.n
        x0 = np.concatenate([centers[:, 0], centers[:, 1], radii])

        # Bounds for SLSQP
        var_bounds = [(0, 1)] * (2 * n) + [(0, 0.5)] * n

        cons_dict = {
            "type": "ineq",
            "fun": self.constraints,
            "jac": self.constraints_jac,
        }

        try:
            # Run Non-Linear Optimization
            res = minimize(
                self.obj_func,
                x0,
                method="SLSQP",
                jac=self.obj_jac,
                constraints=cons_dict,
                bounds=var_bounds,
                options={"maxiter": maxiter, "ftol": 1e-4, "disp": False},
            )

            # Extract centers
            xc = np.clip(res.x[:n], 0, 1)
            yc = np.clip(res.x[n : 2 * n], 0, 1)
            new_centers = np.column_stack((xc, yc))

            # Polish radii strictly with LP
            new_radii, score = self.polish_radii(new_centers)

            # Final Safety Cleanup
            new_radii = self.cleanup_solution(new_centers, new_radii)
            score = np.sum(new_radii)

            return new_centers, new_radii, score

        except Exception:
            return centers, radii, np.sum(radii)

    def perturb(self, centers, magnitude=0.05):
        """
        Apply Gaussian noise to circle centers for basin hopping.
        
        Args:
            centers: (n,2) array of circle centers
            magnitude: Standard deviation of Gaussian noise
            
        Returns:
            array: Perturbed centers clipped to unit square [0,1] range
        """
        noise = np.random.normal(0, magnitude, centers.shape)
        new_centers = centers + noise
        new_centers = np.clip(new_centers, 0, 1)
        return new_centers


def run_packing(num_circles=26):
    """
    Main optimization routine for circle packing problem.
    
    Implements a basin hopping algorithm with:
    - Multiple initialization strategies
    - Local optimization with analytical gradients
    - Linear programming for radius maximization
    - Time-limited execution
    
    Args:
        num_circles (int): Number of circles to pack (default 26)
        
    Returns:
        tuple: (centers, radii, total_radius) where:
            - centers: (n,2) array of optimal circle positions
            - radii: (n,) array of optimized radii
            - total_radius: Sum of all radii
    """
    start_time = time.time()
    time_limit = 950.0

    optimizer = PackingOptimizer(num_circles)

    # Global best tracking
    best_centers = np.zeros((num_circles, 2))
    best_radii = np.zeros(num_circles)
    best_score = -np.inf

    # Basin Hopping Parameters
    max_hops = 25
    perturb_magnitude = 0.05

    attempt = 0

    while time.time() - start_time < time_limit:
        attempt += 1

        # 1. Initialization
        if attempt == 1:
            centers, radii = optimizer.get_initial_guess("ring")
            centers, radii, score = optimizer.local_minimize(
                centers, radii, maxiter=100
            )
        else:
            centers, radii = optimizer.get_initial_guess("random")
            centers, radii, score = optimizer.local_minimize(centers, radii, maxiter=30)

        if score > best_score:
            best_score = score
            best_centers = centers.copy()
            best_radii = radii.copy()

        # 2. Basin Hopping Loop
        current_centers = centers
        current_radii = radii
        current_score = score

        no_improv = 0

        for hop in range(max_hops):
            if time.time() - start_time > time_limit:
                break

            # Perturb
            candidate_centers = optimizer.perturb(
                current_centers, magnitude=perturb_magnitude
            )

            # Minimize (Relax)
            candidate_centers, candidate_radii, candidate_score = (
                optimizer.local_minimize(candidate_centers, current_radii, maxiter=40)
            )

            # Acceptance (Greedy)
            if candidate_score > current_score + 1e-7:
                current_score = candidate_score
                current_centers = candidate_centers
                current_radii = candidate_radii
                no_improv = 0

                if current_score > best_score:
                    best_score = current_score
                    best_centers = current_centers.copy()
                    best_radii = current_radii.copy()
            else:
                no_improv += 1

            if no_improv >= 5:
                break

    # Final cleanup before return
    best_radii = optimizer.cleanup_solution(best_centers, best_radii)
    best_score = np.sum(best_radii)

    return best_centers, best_radii, best_score


# --- Verification Code ---


def _circles_overlap(centers, radii):
    """
    Check if any circles in the configuration overlap.
    
    Args:
        centers: (n,2) array of circle centers
        radii: (n,) array of circle radii
        
    Returns:
        bool: True if any overlapping circles found, False otherwise
    """
    n = centers.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
            if radii[i] + radii[j] > dist:
                return True
    return False


def check_construction(centers, radii, n) -> dict[str, float]:
    """
    Validate circle packing configuration.
    
    Checks:
    - All circles are within unit square
    - No overlapping circles
    - Valid radii values
    
    Args:
        centers: (n,2) array of circle centers
        radii: (n,) array of circle radii
        n: Expected number of circles
        
    Returns:
        dict: Validation results with "sum_of_radii" score
    """
    if centers.shape != (n, 2) or not np.isfinite(centers).all():
        return {"sum_of_radii": -np.inf}

    is_contained = ((radii[:, None] <= centers) & (centers <= 1 - radii[:, None])).all(
        axis=1
    )
    if not is_contained.all():
        return {"sum_of_radii": -np.inf}

    if radii.shape != (n,) or not np.isfinite(radii).all() or not (0 <= radii).all():
        return {"sum_of_radii": -np.inf}

    if _circles_overlap(centers, radii):
        return {"sum_of_radii": -np.inf}

    return {"sum_of_radii": float(np.sum(radii))}


if __name__ == "__main__":
    c, r, s = run_packing(26)
    print(f"Sum: {s}")
    for i in range(len(c)):
        print(f"[{c[i][0]:.16f}, {c[i][1]:.16f}, {r[i]:.16f}],")
    valid = check_construction(c, r, 26)
    print(valid)
