"""Optimal hexagonal packing solution implementation."""

import math
import random
import numpy as np
from scipy.optimize import minimize, basinhopping

# --- Auxiliary Calculation Functions (Provided) ---
EPSILON = 1e-12


def hexagon_vertices(center_x, center_y, side_length, angle_degrees):
    """
    Calculate the vertices of a hexagon.
    """
    vertices = []
    angle_radians = math.radians(angle_degrees)
    for i in range(6):
        angle = angle_radians + 2 * math.pi * i / 6
        x = center_x + side_length * math.cos(angle)
        y = center_y + side_length * math.sin(angle)
        vertices.append((x, y))
    return vertices


def normalize_vector(v):
    """
    Normalize a 2D vector to unit length.
    """
    magnitude = math.sqrt(v[0] ** 2 + v[1] ** 2)
    return (v[0] / magnitude, v[1] / magnitude) if magnitude > EPSILON else (0.0, 0.0)


def get_normals(vertices):
    """
    Calculate normal vectors for all edges of a polygon.
    """
    normals = []
    for i in range(len(vertices)):
        p1 = vertices[i]
        p2 = vertices[(i + 1) % len(vertices)]
        edge = (p2[0] - p1[0], p2[1] - p1[1])
        normal = normalize_vector((-edge[1], edge[0]))
        normals.append(normal)
    return normals


def project_polygon(vertices, axis):
    """
    Project a polygon onto an axis and return the min and max projection values.
    """
    min_proj = float("inf")
    max_proj = float("-inf")
    for vertex in vertices:
        projection = vertex[0] * axis[0] + vertex[1] * axis[1]
        min_proj = min(min_proj, projection)
        max_proj = max(max_proj, projection)
    return min_proj, max_proj


def overlap_1d(min1, max1, min2, max2):
    """
    Check if two 1D intervals overlap (with epsilon tolerance).
    """
    return max1 >= min2 - EPSILON and max2 >= min1 - EPSILON


def polygons_intersect(vertices1, vertices2):
    """
    Check if two convex polygons intersect using the Separating Axis Theorem.
    """
    normals1 = get_normals(vertices1)
    normals2 = get_normals(vertices2)
    axes = normals1 + normals2

    for axis in axes:
        min1, max1 = project_polygon(vertices1, axis)
        min2, max2 = project_polygon(vertices2, axis)
        if not overlap_1d(min1, max1, min2, max2):
            return False
    return True


def hexagons_are_disjoint(hex1_params, hex2_params):
    """
    Check if two hexagons are disjoint (non-overlapping).
    """
    hex1_vertices = hexagon_vertices(*hex1_params)
    hex2_vertices = hexagon_vertices(*hex2_params)
    return not polygons_intersect(hex1_vertices, hex2_vertices)


def is_inside_hexagon(point, hex_params):
    """
    Check if a point is inside a hexagon.
    """
    hex_vertices = hexagon_vertices(*hex_params)
    for i in range(len(hex_vertices)):
        p1 = hex_vertices[i]
        p2 = hex_vertices[(i + 1) % len(hex_vertices)]
        edge_vector = (p2[0] - p1[0], p2[1] - p1[1])
        point_vector = (point[0] - p1[0], point[1] - p1[1])
        cross_product = (
            edge_vector[0] * point_vector[1] - edge_vector[1] * point_vector[0]
        )

        if cross_product < -EPSILON:
            return False
    return True


def all_hexagons_contained(inner_hex_params_list, outer_hex_params):
    """
    Check if all inner hexagons are completely contained within the outer hexagon.
    """
    for inner_hex_params in inner_hex_params_list:
        inner_hex_vertices = hexagon_vertices(*inner_hex_params)
        for vertex in inner_hex_vertices:
            if not is_inside_hexagon(vertex, outer_hex_params):
                return False
    return True


def verify_construction(
    inner_hex_data, outer_hex_center, outer_hex_side_length, outer_hex_angle_degrees
):
    """
    Verify the complete construction meets all requirements.
    """
    inner_hex_params_list = [(x, y, 1, angle) for x, y, angle in inner_hex_data]
    outer_hex_params = (
        outer_hex_center[0],
        outer_hex_center[1],
        outer_hex_side_length,
        outer_hex_angle_degrees,
    )

    if outer_hex_side_length < EPSILON:
        return False

    for i in range(len(inner_hex_params_list)):
        for j in range(i + 1, len(inner_hex_params_list)):
            if not hexagons_are_disjoint(
                inner_hex_params_list[i], inner_hex_params_list[j]
            ):
                return False

    if not all_hexagons_contained(inner_hex_params_list, outer_hex_params):
        return False

    return True


# --- Optimization Classes and Functions ---


class AnalyticPacking:
    """
    Analytic packing optimizer for hexagonal packing optimization problem.
    
    This class implements an analytic approach to optimize the packing of multiple
    hexagons inside a larger hexagon. It uses gradient-based optimization with
    custom energy functions for boundary constraints and overlap prevention.
    
    Attributes:
        n (int): Number of inner hexagons to pack
        base_verts (numpy.ndarray): Base vertices of a standard hexagon
        base_normals (numpy.ndarray): Normal vectors of the standard hexagon
        outer_normals (numpy.ndarray): Normal vectors for the outer hexagon
    """
    def __init__(self, n_hex=11):
        self.n = n_hex
        # Base vertices for a standard hexagon (radius 1, angle 0)
        angles = np.deg2rad(np.array([0, 60, 120, 180, 240, 300]))
        self.base_verts = np.stack([np.cos(angles), np.sin(angles)], axis=1)  # (6, 2)

        # Normals of the standard hexagon
        norm_angles = np.deg2rad(np.array([30, 90, 150, 210, 270, 330]))
        self.base_normals = np.stack(
            [np.cos(norm_angles), np.sin(norm_angles)], axis=1
        )  # (6, 2)

        # Outer hexagon normals (fixed, angle 0)
        self.outer_normals = self.base_normals.copy()

    def get_vertices_and_jacobian(self, flat_params):
        """
        Compute vertices and their derivatives w.r.t (x, y, theta).
        """
        data = flat_params.reshape(self.n, 3)
        xc, yc, theta = data[:, 0], data[:, 1], data[:, 2]

        rads = np.deg2rad(theta)
        cos_t = np.cos(rads)
        sin_t = np.sin(rads)

        v_bx = self.base_verts[:, 0]  # (6,)
        v_by = self.base_verts[:, 1]  # (6,)

        # Expand for N
        term1_x = cos_t[:, None] * v_bx[None, :]
        term2_x = sin_t[:, None] * v_by[None, :]

        term1_y = sin_t[:, None] * v_bx[None, :]
        term2_y = cos_t[:, None] * v_by[None, :]

        vx = xc[:, None] + term1_x - term2_x
        vy = yc[:, None] + term1_y + term2_y

        verts = np.stack([vx, vy], axis=2)  # (N, 6, 2)

        d_rad = np.pi / 180.0

        d_vx = (
            -sin_t[:, None] * v_bx[None, :] - cos_t[:, None] * v_by[None, :]
        ) * d_rad
        d_vy = (cos_t[:, None] * v_bx[None, :] - sin_t[:, None] * v_by[None, :]) * d_rad

        d_verts_theta = np.stack([d_vx, d_vy], axis=2)  # (N, 6, 2)

        return verts, d_verts_theta

    def energy_and_grad(self, flat_params, outer_side_length):
        """
        Compute Total Energy and Gradient.
        """
        N = self.n
        grad = np.zeros_like(flat_params)

        verts, d_verts_theta = self.get_vertices_and_jacobian(flat_params)

        # 1. Boundary Energy
        apothem = outer_side_length * 0.86602540378

        projections = np.tensordot(verts, self.outer_normals.T, axes=([2], [0]))

        violations = projections - apothem
        mask = violations > 0

        energy_boundary = np.sum(violations[mask] ** 2)

        # Gradient Boundary
        for i in range(N):
            coeffs = 2 * violations[i] * mask[i]  # (6, 6)
            grad_cx = np.sum(coeffs * self.outer_normals[:, 0][None, :])
            grad_cy = np.sum(coeffs * self.outer_normals[:, 1][None, :])

            grad[3 * i] += grad_cx
            grad[3 * i + 1] += grad_cy

            d_proj = np.dot(d_verts_theta[i], self.outer_normals.T)
            grad_theta = np.sum(coeffs * d_proj)
            grad[3 * i + 2] += grad_theta

        # 2. Overlap Energy (Soft SAT)
        data = flat_params.reshape(N, 3)
        angles = data[:, 2]  # (N,)

        rads = np.deg2rad(angles)
        cos_t = np.cos(rads)[:, None]
        sin_t = np.sin(rads)[:, None]

        nx = self.base_normals[:, 0]
        ny = self.base_normals[:, 1]

        axes_x = nx[None, :] * cos_t - ny[None, :] * sin_t
        axes_y = nx[None, :] * sin_t + ny[None, :] * cos_t
        all_axes = np.stack([axes_x, axes_y], axis=2)  # (N, 6, 2)

        d_rad = np.pi / 180.0
        d_axes_x = (-nx[None, :] * sin_t - ny[None, :] * cos_t) * d_rad
        d_axes_y = (nx[None, :] * cos_t - ny[None, :] * sin_t) * d_rad
        d_all_axes = np.stack([d_axes_x, d_axes_y], axis=2)  # (N, 6, 2)

        energy_overlap = 0.0

        # Helper to get grad of dot product v.n
        def get_dot_grads(
            v_idx, is_min, obj_idx, other_idx, ax, d_ax_dtheta_owner, axis_is_owner
        ):
            g_c = ax
            g_th = np.dot(d_verts_theta[obj_idx, v_idx], ax)
            if axis_is_owner:
                v_pos = verts[obj_idx, v_idx]
                term = np.dot(v_pos, d_ax_dtheta_owner)
                g_th += term
            return g_c, g_th

        for i in range(N):
            for j in range(i + 1, N):
                dx = data[i, 0] - data[j, 0]
                dy = data[i, 1] - data[j, 1]
                if dx * dx + dy * dy > 4.5:
                    continue

                current_axes = np.concatenate(
                    [all_axes[i], all_axes[j]], axis=0
                )  # (12, 2)

                p_i = np.dot(verts[i], current_axes.T)
                p_j = np.dot(verts[j], current_axes.T)

                min_i = np.min(p_i, axis=0)
                max_i = np.max(p_i, axis=0)
                min_j = np.min(p_j, axis=0)
                max_j = np.max(p_j, axis=0)

                sep1 = min_i - max_j
                sep2 = min_j - max_i

                sep = np.maximum(sep1, sep2)  # (12,)

                max_sep = np.max(sep)
                best_axis_idx = np.argmax(sep)

                if max_sep >= -1e-6:
                    continue

                energy_overlap += max_sep**2

                k = best_axis_idx
                axis = current_axes[k]

                axis_from_i = k < 6
                local_axis_idx = k if axis_from_i else k - 6

                is_sep1 = sep1[k] > sep2[k]
                factor = 2 * max_sep

                idx_min_i = np.argmin(p_i[:, k])
                idx_max_i = np.argmax(p_i[:, k])
                idx_min_j = np.argmin(p_j[:, k])
                idx_max_j = np.argmax(p_j[:, k])

                d_axis = d_all_axes[i if axis_from_i else j, local_axis_idx]

                if is_sep1:
                    gc_i, gt_i = get_dot_grads(
                        idx_min_i, True, i, j, axis, d_axis, axis_from_i
                    )
                    gc_j, gt_j = get_dot_grads(
                        idx_max_j, False, j, i, axis, d_axis, not axis_from_i
                    )

                    grad[3 * i] += factor * gc_i[0]
                    grad[3 * i + 1] += factor * gc_i[1]
                    grad[3 * i + 2] += factor * gt_i

                    grad[3 * j] -= factor * gc_j[0]
                    grad[3 * j + 1] -= factor * gc_j[1]
                    grad[3 * j + 2] -= factor * gt_j

                    if not axis_from_i:
                        term = np.dot(verts[i, idx_min_i], d_axis)
                        grad[3 * j + 2] += factor * term
                    if axis_from_i:
                        term = np.dot(verts[j, idx_max_j], d_axis)
                        grad[3 * i + 2] -= factor * term

                else:
                    gc_j, gt_j = get_dot_grads(
                        idx_min_j, True, j, i, axis, d_axis, not axis_from_i
                    )
                    gc_i, gt_i = get_dot_grads(
                        idx_max_i, False, i, j, axis, d_axis, axis_from_i
                    )

                    grad[3 * j] += factor * gc_j[0]
                    grad[3 * j + 1] += factor * gc_j[1]
                    grad[3 * j + 2] += factor * gt_j

                    grad[3 * i] -= factor * gc_i[0]
                    grad[3 * i + 1] -= factor * gc_i[1]
                    grad[3 * i + 2] -= factor * gt_i

                    if axis_from_i:
                        term = np.dot(verts[j, idx_min_j], d_axis)
                        grad[3 * i + 2] += factor * term
                    if not axis_from_i:
                        term = np.dot(verts[i, idx_max_i], d_axis)
                        grad[3 * j + 2] -= factor * term

        return energy_boundary * 5000 + energy_overlap * 20000, grad


class TopologicalPerturbation:
    """
    Topological perturbation for basin hopping optimization.
    
    This class provides perturbation strategies to help escape local minima
    during optimization. It implements three types of perturbations:
    - Jiggle: Small random perturbations to positions and angles
    - Rotate: Significant rotations of individual hexagons
    - Swap: Exchanging positions of pairs of hexagons
    
    Attributes:
        n (int): Number of hexagons in the system
    """
    def __init__(self, n_hex=11):
        self.n = n_hex

    def __call__(self, x):
        """
        Apply topological perturbation to the current solution.
        
        This method implements the callable interface for basin hopping.
        It randomly selects one of three perturbation strategies to help
        escape local minima during optimization.
        
        Args:
            x (numpy.ndarray): Current parameter vector of shape (33,) containing
                              [x1, y1, angle1, x2, y2, angle2, ...] for 11 hexagons
        
        Returns:
            numpy.ndarray: Perturbed parameter vector
        """
        x_new = x.copy()

        # Moves: Jiggle (30%), Rotate (35%), Swap (35%)
        # Increased swap/rotate prob slightly
        r = random.random()

        if r < 0.3:
            # Jiggle
            noise = np.random.normal(0, 0.05, size=x.shape)
            noise[2::3] *= 0.2
            x_new += noise
        elif r < 0.65:
            # Rotate
            idx = random.randint(0, self.n - 1)
            # Add +/- 30 or 60 degrees
            angle_step = random.choice([-60, -30, 30, 60])
            x_new[3 * idx + 2] += angle_step
        else:
            # Swap
            i, j = random.sample(range(self.n), 2)
            block_i = x_new[3 * i : 3 * i + 3].copy()
            block_j = x_new[3 * j : 3 * j + 3].copy()
            x_new[3 * i : 3 * i + 3] = block_j
            x_new[3 * j : 3 * j + 3] = block_i

        return x_new


def get_spiral_seed(n=11):
    """
    Generate initial seed configuration using spiral pattern.
    
    This function creates a spiral arrangement of hexagon centers based on
    hexagonal grid coordinates. It uses breadth-first search to generate
    coordinates in a spiral pattern from the center outward.
    
    Args:
        n (int): Number of hexagons to generate coordinates for
        
    Returns:
        numpy.ndarray: Parameter vector containing [x1, y1, angle1, x2, y2, angle2, ...]
                      for n hexagons, with all angles set to 0.0
    """
    d = math.sqrt(3)
    coords = [(0, 0)]
    for i in range(6):
        theta = i * math.pi / 3
        coords.append((d * math.cos(theta), d * math.sin(theta)))

    v1 = (math.sqrt(3), 0)
    v2 = (math.sqrt(3) / 2, 1.5)

    visited = set([(0, 0)])
    queue = [(0, 0)]
    grid_coords = []

    while len(grid_coords) < n:
        if not queue:
            break
        q, r = queue.pop(0)
        px = q * v1[0] + r * v2[0]
        py = q * v1[1] + r * v2[1]
        grid_coords.append((px, py))
        neighbors = [
            (q + 1, r),
            (q - 1, r),
            (q, r + 1),
            (q, r - 1),
            (q + 1, r - 1),
            (q - 1, r + 1),
        ]
        neighbors.sort(
            key=lambda p: (p[0] * v1[0] + p[1] * v2[0]) ** 2
            + (p[0] * v1[1] + p[1] * v2[1]) ** 2
        )
        for nq, nr in neighbors:
            if (nq, nr) not in visited:
                visited.add((nq, nr))
                queue.append((nq, nr))

    params = []
    for i in range(n):
        if i < len(grid_coords):
            params.extend([grid_coords[i][0], grid_coords[i][1], 0.0])
        else:
            params.extend([0, 0, 0])
    return np.array(params)


def get_row_seed(n=11):
    """
    Generate initial seed configuration using row pattern.
    
    This function creates a rectangular arrangement of hexagon centers
    with a 3-4-4 row configuration (bottom row: 3 hexagons, middle: 4, top: 4).
    This pattern mimics common hexagonal packing arrangements.
    
    Args:
        n (int): Number of hexagons to generate coordinates for (should be 11)
        
    Returns:
        numpy.ndarray: Parameter vector containing [x1, y1, angle1, x2, y2, angle2, ...]
                      for n hexagons, with all angles set to 0.0
    """
    # Try 3-4-4
    rows = [3, 4, 4]
    coords = []
    d = math.sqrt(3)
    y_offset = -1.5  # approx center

    for r_idx, count in enumerate(rows):
        y = y_offset + r_idx * 1.5
        x_start = -(count - 1) * d / 2.0
        for c_idx in range(count):
            x = x_start + c_idx * d
            coords.append((x, y))

    params = []
    for i in range(n):
        if i < len(coords):
            params.extend([coords[i][0], coords[i][1], 0.0])
        else:
            params.extend([0, 0, 0])
    return np.array(params)


def optimize_construct():
    """
    Optimize the construction of hexagonal packing configuration.
    
    This function implements the main optimization pipeline for packing 11 hexagons
    inside a larger hexagon. It uses a combination of:
    - Seed initialization (spiral and row patterns)
    - Gradient-based optimization (L-BFGS-B)
    - Basin hopping with topological perturbations
    - Final polish and constraint enforcement
    
    Returns:
        tuple: (inner_hex_data, outer_hex, final_side, outer_angle)
            - inner_hex_data: List of [x, y, angle] for each inner hexagon
            - outer_hex: Outer hexagon center coordinates [x, y]
            - final_side: Side length of the outer hexagon
            - outer_angle: Angle of the outer hexagon (always 0.0)
    """
    optimizer = AnalyticPacking(11)
    perturber = TopologicalPerturbation(11)

    # Init Seeds
    seeds = [get_spiral_seed(11), get_row_seed(11)]
    best_start_energy = float("inf")
    current_params = seeds[0]

    bounds = []
    for _ in range(11):
        bounds.append((-10.0, 10.0))
        bounds.append((-10.0, 10.0))
        bounds.append((None, None))

    start_side = 4.1

    # Pick best seed
    print("Selecting best seed...")
    for s in seeds:
        res = minimize(
            optimizer.energy_and_grad,
            s,
            args=(start_side,),
            method="L-BFGS-B",
            jac=True,
            bounds=bounds,
        )
        if res.fun < best_start_energy:
            best_start_energy = res.fun
            current_params = res.x

    current_side = start_side
    step = 0.05
    min_step = 0.0002  # Go very fine at the end

    # Wrapper to intercept calls in Basin Hopping
    class ObjectiveWrapper:
        """
        Wrapper class for basin hopping optimization to track valid solutions.
        
        This wrapper intercepts optimization calls to detect when a valid solution
        (energy < 1e-8) is found during the basin hopping process.
        
        Attributes:
            func (callable): The original objective function to wrap
            args (tuple): Additional arguments for the objective function
            found_valid (bool): Flag indicating if a valid solution was found
            best_x (numpy.ndarray): Parameters of the best valid solution found
        """
        def __init__(self, func, args):
            """
            Initialize the wrapper with objective function and arguments.
            
            Args:
                func (callable): The objective function to wrap
                args (tuple): Additional arguments for the objective function
            """
            self.func = func
            self.args = args
            self.found_valid = False
            self.best_x = None

        def __call__(self, x):
            """
            Call the wrapped objective function and track valid solutions.
            
            Args:
                x (numpy.ndarray): Current parameter vector
                
            Returns:
                tuple: (energy, gradient) from the objective function
            """
            e, g = self.func(x, *self.args)
            if e < 1e-8:  # Stricter tolerance
                self.found_valid = True
                self.best_x = x
            return e, g

    while step >= min_step:
        target_side = current_side - step
        if target_side < 3.8:
            break

        print(f"Targeting side: {target_side:.4f}")

        success = False

        # Try 1: Quick minimize from current
        res_quick = minimize(
            optimizer.energy_and_grad,
            current_params,
            args=(target_side,),
            method="L-BFGS-B",
            jac=True,
            bounds=bounds,
            options={"ftol": 1e-8, "gtol": 1e-8},
        )
        if res_quick.fun < 1e-8:
            current_params = res_quick.x
            success = True
        else:
            # Try 2: Basin Hopping
            wrapper = ObjectiveWrapper(optimizer.energy_and_grad, (target_side,))

            minimizer_kwargs = {
                "method": "L-BFGS-B",
                "jac": True,
                "bounds": bounds,
                "options": {"ftol": 1e-8, "gtol": 1e-8},
            }

            # Dynamic iterations
            if target_side < 3.95:
                n_iter = 300  # Grind harder
            else:
                n_iter = 100

            try:
                res_bh = basinhopping(
                    wrapper,  # Use wrapper
                    current_params,
                    niter=n_iter,
                    T=0.005,  # Lower temperature to stay deep
                    take_step=perturber,
                    minimizer_kwargs=minimizer_kwargs,
                    stepsize=0.1,
                    callback=lambda x, f, acc: wrapper.found_valid,
                )
            except:
                pass

            if wrapper.found_valid:
                current_params = wrapper.best_x
                success = True
            elif res_bh.fun < 1e-8:
                current_params = res_bh.x
                success = True

        if success:
            current_side = target_side
            print(f"  Success. Current side: {current_side:.4f}")
        else:
            print(f"  Failed. Backtracking.")
            step *= 0.5

    # Final Polish
    print("Final Polish...")
    res_final = minimize(
        optimizer.energy_and_grad,
        current_params,
        args=(current_side,),
        method="L-BFGS-B",
        jac=True,
        bounds=bounds,
        options={"ftol": 1e-14, "gtol": 1e-14, "maxiter": 5000},
    )
    current_params = res_final.x

    # Separation Pass (Strict enforcement)
    data = current_params.reshape(11, 3)
    inner_hex_data = []
    for row in data:
        inner_hex_data.append([row[0], row[1], row[2] % 360])

    params_list = [(h[0], h[1], 1.0, h[2]) for h in inner_hex_data]

    # Push loop to resolve epsilon overlaps
    for _ in range(5000):
        any_moved = False
        for i in range(11):
            for j in range(i + 1, 11):
                if not hexagons_are_disjoint(params_list[i], params_list[j]):
                    any_moved = True
                    p1 = params_list[i]
                    p2 = params_list[j]
                    dx = p1[0] - p2[0]
                    dy = p1[1] - p2[1]
                    d = math.sqrt(dx * dx + dy * dy)
                    if d < 1e-9:
                        dx, dy = random.random() - 0.5, random.random() - 0.5
                        d = math.sqrt(dx * dx + dy * dy)

                    push = 1e-6  # Very tiny push
                    nx, ny = dx / d, dy / d

                    params_list[i] = (p1[0] + nx * push, p1[1] + ny * push, 1.0, p1[3])
                    params_list[j] = (p2[0] - nx * push, p2[1] - ny * push, 1.0, p2[3])

        if not any_moved:
            break

    final_hex_data = [[p[0], p[1], p[3]] for p in params_list]

    # Shift to positive coordinates
    shift_x, shift_y = 10.0, 10.0
    for h in final_hex_data:
        h[0] += shift_x
        h[1] += shift_y

    outer_center = [shift_x, shift_y]
    outer_angle = 0.0

    # Recalculate side length to perfectly enclose everything
    max_apothem = 0.0
    outer_normals_angles = [30, 90, 150, 210, 270, 330]

    for h in final_hex_data:
        vs = hexagon_vertices(h[0], h[1], 1.0, h[2])
        for vx, vy in vs:
            rx = vx - outer_center[0]
            ry = vy - outer_center[1]
            for ang_deg in outer_normals_angles:
                ang = math.radians(ang_deg)
                nx, ny = math.cos(ang), math.sin(ang)
                proj = rx * nx + ry * ny
                max_apothem = max(max_apothem, proj)

    final_side = max_apothem * 2.0 / math.sqrt(3) + 1e-9

    return final_hex_data, outer_center, final_side, outer_angle


if __name__ == "__main__":
    inner_hex_data, outer_hex, outer_side_length, outer_angle = optimize_construct()
    print(f"outer_side_length: {outer_side_length:.16f}")
    print(f"outer_angle: {outer_angle:.16f}")
    for i in range(len(inner_hex_data)):
        print(
            f"[{inner_hex_data[i][0]:.16f}, {inner_hex_data[i][1]:.16f}, {inner_hex_data[i][2]:.16f}],"
        )
    print(f"outer_hex_x: {outer_hex[0]:.16f} outer_hex_y: {outer_hex[1]:.16f}")
    print(outer_hex)

    valid = verify_construction(
        inner_hex_data, outer_hex, outer_side_length, outer_angle
    )
    print(f"valid: {valid}")
