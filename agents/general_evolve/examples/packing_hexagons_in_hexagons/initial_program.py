"""
Optimized Packing of 11 Unit Hexagons - Advanced Hybrid Algorithm with Physics Simulation
"""

# EVOLVE-BLOCK-START

import math
import random
import numpy as np
from typing import List, Tuple

# Constants
EPSILON = 1e-12
SQRT3 = math.sqrt(3.0)

def optimize_construct():
    """
    Optimized packing of 11 unit hexagons using advanced hybrid algorithm.
    Combines genetic algorithm, physics simulation, and pattern optimization.
    
    Returns:
        inner_hex_data: List of [x, y, angle_degrees] for inner hexagons.
        outer_hex_center: [x, y] for the outer hexagon center.
        outer_hex_side_length: Side length of the outer hexagon.
        outer_hex_angle_degrees: Rotation angle of the outer hexagon in degrees.
    """
    
    # Main optimization pipeline
    best_solution = None
    best_side_length = float('inf')
    
    # Run multiple optimization attempts with different strategies
    for attempt in range(3):
        if attempt == 0:
            # Pattern-based optimization
            solution = pattern_based_optimization()
        elif attempt == 1:
            # Physics-based optimization
            solution = physics_based_optimization()
        else:
            # Hybrid optimization
            solution = hybrid_optimization()
        
        # Extract side length
        config, outer_center, side_length, outer_angle = solution
        
        # Verify the solution
        inner_hex_data = []
        for x, y, angle in config:
            inner_hex_data.append([np.float64(x), np.float64(y), np.float64(angle)])
        
        # Test verification with tighter bounds
        if side_length < best_side_length:
            if verify_construction(inner_hex_data, 
                                  [np.float64(outer_center[0]), np.float64(outer_center[1])], 
                                  np.float64(side_length), 
                                  np.float64(outer_angle)):
                best_solution = solution
                best_side_length = side_length
        
        # Early exit if we achieve target
        if best_side_length < 3.931:
            break
    
    # Final refinement
    if best_solution:
        config, outer_center, side_length, outer_angle = best_solution
        config, side_length = final_optimization(config, side_length)
    else:
        # Fallback: generate a simple valid solution
        config, side_length = generate_simple_packing()
        outer_center, outer_angle = optimize_outer_hexagon(config, side_length)
    
    # Format output
    inner_hex_data = []
    for x, y, angle in config:
        inner_hex_data.append([
            np.float64(x),
            np.float64(y),
            np.float64(angle)
        ])
    
    outer_hex_center = [
        np.float64(outer_center[0]),
        np.float64(outer_center[1])
    ]
    
    outer_hex_side_length = np.float64(side_length)
    outer_hex_angle_degrees = np.float64(outer_angle)
    
    return inner_hex_data, outer_hex_center, outer_hex_side_length, outer_hex_angle_degrees


def pattern_based_optimization():
    """Optimization based on hexagonal packing patterns."""
    
    # Generate multiple promising patterns
    patterns = []
    
    # Pattern 1: Optimized concentric rings (mathematically derived)
    pattern1 = []
    # Center hexagon
    pattern1.append((0.0, 0.0, 0.0))
    
    # First ring - 6 hexagons
    ring1_radius = SQRT3 + 0.01  # Slightly increased for better separation
    for i in range(6):
        angle = 60.0 * i
        x = ring1_radius * math.cos(math.radians(angle))
        y = ring1_radius * math.sin(math.radians(angle))
        # Alternate rotations for tighter packing
        rotation = 30.0 if i % 2 == 0 else 0.0
        pattern1.append((x, y, rotation))
    
    # Additional 4 hexagons in optimized positions
    # Place in the most promising gaps
    gap_positions = [
        (30.0, 2.0), (90.0, 2.0), (150.0, 2.0), (210.0, 2.0)
    ]
    for i, (angle, radius) in enumerate(gap_positions[:4]):
        x = radius * math.cos(math.radians(angle))
        y = radius * math.sin(math.radians(angle))
        rotation = 30.0 if i % 2 == 1 else 0.0
        pattern1.append((x, y, rotation))
    
    patterns.append(pattern1)
    
    # Pattern 2: Offset rows with optimized spacing
    pattern2 = []
    # Three-row structure: 3-4-4
    row_spacing = SQRT3 * 0.5
    col_spacing = 1.5
    
    # Bottom row: 3 hexagons
    for i in range(3):
        x = col_spacing * (i - 1)
        y = -row_spacing
        pattern2.append((x, y, 0.0))
    
    # Middle row: 4 hexagons (offset)
    for i in range(4):
        x = col_spacing * (i - 1.5)
        y = 0.0
        pattern2.append((x, y, 30.0))
    
    # Top row: 4 hexagons
    for i in range(4):
        x = col_spacing * (i - 1.5)
        y = row_spacing
        pattern2.append((x, y, 0.0))
    
    patterns.append(pattern2)
    
    # Pattern 3: Circular cluster with varied rotations
    pattern3 = []
    center_positions = [
        (0.0, 0.0, 0.0),
        (1.5, 0.0, 30.0),
        (0.75, SQRT3 * 0.5, 0.0),
        (0.75, -SQRT3 * 0.5, 30.0),
        (2.25, SQRT3 * 0.5, 0.0),
        (2.25, -SQRT3 * 0.5, 30.0),
        (0.0, SQRT3, 0.0),
        (1.5, SQRT3, 30.0),
        (0.0, -SQRT3, 30.0),
        (1.5, -SQRT3, 0.0),
        (3.0, 0.0, 0.0)
    ]
    pattern3 = center_positions
    patterns.append(pattern3)
    
    # Optimize each pattern
    best_config = None
    best_side = float('inf')
    best_outer = (0.0, 0.0)
    best_angle = 0.0
    
    for pattern in patterns:
        # Refine pattern
        config = refine_pattern(pattern)
        
        # Remove any overlaps
        config = resolve_all_overlaps(config)
        
        # Optimize outer hexagon
        side_length = find_minimal_side(config)
        outer_center, outer_angle = optimize_outer_hexagon(config, side_length)
        
        if side_length < best_side:
            # Verify the configuration
            if verify_configuration(config, outer_center, side_length, outer_angle):
                best_config = config
                best_side = side_length
                best_outer = outer_center
                best_angle = outer_angle
    
    return best_config, best_outer, best_side, best_angle


def physics_based_optimization():
    """Physics-inspired optimization with repulsion forces."""
    
    # Initialize with random configuration
    config = []
    for _ in range(11):
        x = random.uniform(-2.0, 2.0)
        y = random.uniform(-2.0, 2.0)
        angle = random.uniform(0.0, 60.0)
        config.append((x, y, angle))
    
    # Physics simulation parameters
    k_repel = 1.0  # Repulsion strength
    k_boundary = 0.5  # Boundary attraction
    damping = 0.95
    dt = 0.1
    
    # Simulate for multiple iterations
    for iteration in range(1000):
        velocities = [(0.0, 0.0, 0.0) for _ in range(11)]
        
        # Calculate repulsion forces
        for i in range(11):
            for j in range(i + 1, 11):
                xi, yi, ai = config[i]
                xj, yj, aj = config[j]
                
                # Check for overlap
                params_i = (xi, yi, 1.0, ai)
                params_j = (xj, yj, 1.0, aj)
                
                if not hexagons_are_disjoint(params_i, params_j):
                    # Calculate repulsion force
                    dx = xi - xj
                    dy = yi - yj
                    dist = max(math.hypot(dx, dy), EPSILON)
                    
                    force_mag = k_repel / (dist * dist)
                    fx = dx / dist * force_mag
                    fy = dy / dist * force_mag
                    
                    # Apply forces
                    vxi, vyi, vai = velocities[i]
                    velocities[i] = (vxi + fx, vyi + fy, vai)
                    
                    vxj, vyj, vaj = velocities[j]
                    velocities[j] = (vxj - fx, vyj - fy, vaj)
        
        # Apply boundary forces (attract toward center)
        center_x = sum(x for x, y, a in config) / 11
        center_y = sum(y for x, y, a in config) / 11
        
        for i in range(11):
            xi, yi, ai = config[i]
            dx = center_x - xi
            dy = center_y - yi
            dist = max(math.hypot(dx, dy), EPSILON)
            
            force_mag = k_boundary * dist
            fx = dx / dist * force_mag
            fy = dy / dist * force_mag
            
            vxi, vyi, vai = velocities[i]
            velocities[i] = (vxi + fx, vyi + fy, vai)
        
        # Update positions with damping
        new_config = []
        for i in range(11):
            xi, yi, ai = config[i]
            vxi, vyi, vai = velocities[i]
            
            new_x = xi + vxi * dt
            new_y = yi + vyi * dt
            new_angle = (ai + vai * dt * 0.1) % 60.0
            
            # Apply damping
            velocities[i] = (vxi * damping, vyi * damping, vai * damping)
            
            new_config.append((new_x, new_y, new_angle))
        
        config = new_config
        
        # Resolve overlaps every 100 iterations
        if iteration % 100 == 0:
            config = resolve_all_overlaps(config)
    
    # Final optimization
    config = resolve_all_overlaps(config)
    side_length = find_minimal_side(config)
    outer_center, outer_angle = optimize_outer_hexagon(config, side_length)
    
    return config, outer_center, side_length, outer_angle


def hybrid_optimization():
    """Hybrid optimization combining pattern and physics approaches."""
    
    # Start with pattern-based solution
    config, outer_center, side_length, outer_angle = pattern_based_optimization()
    
    if config is None:
        # Fallback to physics-based
        return physics_based_optimization()
    
    # Refine with local search
    for _ in range(500):
        # Try small perturbations
        candidate = []
        for x, y, angle in config:
            dx = random.uniform(-0.02, 0.02)
            dy = random.uniform(-0.02, 0.02)
            dangle = random.uniform(-1.0, 1.0)
            candidate.append((x + dx, y + dy, (angle + dangle) % 60.0))
        
        # Check if candidate is better
        candidate = resolve_all_overlaps(candidate)
        cand_side = find_minimal_side(candidate)
        
        if cand_side < side_length:
            config = candidate
            side_length = cand_side
    
    # Final outer hexagon optimization
    outer_center, outer_angle = optimize_outer_hexagon(config, side_length)
    
    return config, outer_center, side_length, outer_angle


def refine_pattern(pattern):
    """Refine a pattern through small adjustments."""
    best_pattern = pattern[:]
    best_score = evaluate_pattern(pattern)
    
    for _ in range(100):
        # Create perturbed pattern
        new_pattern = []
        for x, y, angle in pattern:
            dx = random.uniform(-0.05, 0.05)
            dy = random.uniform(-0.05, 0.05)
            dangle = random.uniform(-5.0, 5.0)
            new_pattern.append((x + dx, y + dy, (angle + dangle) % 60.0))
        
        new_score = evaluate_pattern(new_pattern)
        
        if new_score < best_score:
            best_pattern = new_pattern
            best_score = new_score
    
    return best_pattern


def evaluate_pattern(pattern):
    """Evaluate the quality of a pattern."""
    if not is_configuration_valid(pattern):
        return float('inf')
    
    # Estimate bounding hexagon size
    all_vertices = []
    for x, y, angle in pattern:
        vertices = hexagon_vertices(x, y, 1.0, angle)
        all_vertices.extend(vertices)
    
    # Find maximum distance from center
    center_x = sum(v[0] for v in all_vertices) / len(all_vertices)
    center_y = sum(v[1] for v in all_vertices) / len(all_vertices)
    
    max_dist = 0.0
    for vx, vy in all_vertices:
        dist = math.hypot(vx - center_x, vy - center_y)
        max_dist = max(max_dist, dist)
    
    return max_dist


def is_configuration_valid(config):
    """Check if configuration has no overlaps."""
    for i in range(len(config)):
        for j in range(i + 1, len(config)):
            params_i = (config[i][0], config[i][1], 1.0, config[i][2])
            params_j = (config[j][0], config[j][1], 1.0, config[j][2])
            
            if not hexagons_are_disjoint(params_i, params_j):
                return False
    return True


def resolve_all_overlaps(config):
    """Resolve all overlaps in configuration."""
    for _ in range(100):  # Max iterations
        has_overlap = False
        
        for i in range(11):
            for j in range(i + 1, 11):
                params_i = (config[i][0], config[i][1], 1.0, config[i][2])
                params_j = (config[j][0], config[j][1], 1.0, config[j][2])
                
                if not hexagons_are_disjoint(params_i, params_j):
                    has_overlap = True
                    
                    # Calculate separation vector
                    xi, yi, ai = config[i]
                    xj, yj, aj = config[j]
                    
                    dx = xi - xj
                    dy = yi - yj
                    dist = max(math.hypot(dx, dy), EPSILON)
                    
                    # Move hexagons apart
                    separation = 0.05
                    sep_x = dx / dist * separation
                    sep_y = dy / dist * separation
                    
                    config[i] = (xi + sep_x, yi + sep_y, ai)
                    config[j] = (xj - sep_x, yj - sep_y, aj)
        
        if not has_overlap:
            break
    
    return config


def find_minimal_side(config):
    """Find minimal side length that contains all hexagons."""
    # Binary search for minimal side
    lower = 0.0
    upper = 10.0  # Initial upper bound
    
    for _ in range(50):  # High precision
        mid = (lower + upper) / 2
        
        # Test if configuration fits in hexagon with side length mid
        if can_fit_in_hexagon(config, mid):
            upper = mid
        else:
            lower = mid
    
    return upper


def can_fit_in_hexagon(config, side_length):
    """Check if configuration fits in hexagon with given side length."""
    # Try multiple outer hexagon orientations
    for outer_angle in [0.0, 15.0, 30.0, 45.0]:
        outer_params = (0.0, 0.0, side_length, outer_angle)
        
        all_inside = True
        for x, y, angle in config:
            inner_vertices = hexagon_vertices(x, y, 1.0, angle)
            
            for vertex in inner_vertices:
                if not is_inside_hexagon(vertex, outer_params):
                    all_inside = False
                    break
            
            if not all_inside:
                break
        
        if all_inside:
            return True
    
    return False


def optimize_outer_hexagon(config, target_side):
    """Find optimal outer hexagon parameters."""
    best_center = (0.0, 0.0)
    best_angle = 0.0
    best_margin = float('inf')
    
    # Try different orientations
    for angle in [0.0, 15.0, 30.0, 45.0]:
        # Find optimal center for this orientation
        center = find_best_center(config, target_side, angle)
        
        # Check margin (distance from vertices to boundary)
        margin = calculate_margin(config, center, target_side, angle)
        
        if margin > -EPSILON and margin < best_margin:
            best_center = center
            best_angle = angle
            best_margin = margin
    
    return best_center, best_angle


def find_best_center(config, side_length, outer_angle):
    """Find best center position for outer hexagon."""
    # Get all vertices
    all_vertices = []
    for x, y, angle in config:
        vertices = hexagon_vertices(x, y, 1.0, angle)
        all_vertices.extend(vertices)
    
    # Simple approach: use centroid
    center_x = sum(v[0] for v in all_vertices) / len(all_vertices)
    center_y = sum(v[1] for v in all_vertices) / len(all_vertices)
    
    # Adjust to ensure all vertices are inside
    outer_params = (center_x, center_y, side_length, outer_angle)
    
    # Check if all vertices are inside
    all_inside = True
    for vertex in all_vertices:
        if not is_inside_hexagon(vertex, outer_params):
            all_inside = False
            break
    
    if not all_inside:
        # Find bounding box and use its center
        min_x = min(v[0] for v in all_vertices)
        max_x = max(v[0] for v in all_vertices)
        min_y = min(v[1] for v in all_vertices)
        max_y = max(v[1] for v in all_vertices)
        
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2
    
    return (center_x, center_y)


def calculate_margin(config, center, side_length, angle):
    """Calculate minimum distance from vertices to boundary."""
    outer_params = (center[0], center[1], side_length, angle)
    
    min_margin = float('inf')
    
    for x, y, inner_angle in config:
        vertices = hexagon_vertices(x, y, 1.0, inner_angle)
        
        for vertex in vertices:
            # Check if vertex is inside
            if not is_inside_hexagon(vertex, outer_params):
                return -float('inf')  # Vertex outside
            
            # Calculate distance to boundary (simplified)
            # For hexagon centered at origin with radius = side_length
            dist_to_center = math.hypot(vertex[0] - center[0], vertex[1] - center[1])
            margin = side_length - dist_to_center
            min_margin = min(min_margin, margin)
    
    return min_margin


def verify_configuration(config, outer_center, side_length, outer_angle):
    """Verify configuration meets all constraints."""
    inner_hex_data = []
    for x, y, angle in config:
        inner_hex_data.append([x, y, angle])
    
    return verify_construction(inner_hex_data, outer_center, side_length, outer_angle)


def final_optimization(config, current_side):
    """Final optimization pass."""
    best_config = config[:]
    best_side = current_side
    
    for _ in range(200):
        # Try to compress configuration
        scale_factor = 0.99
        new_config = []
        
        # Find center
        center_x = sum(x for x, y, a in config) / 11
        center_y = sum(y for x, y, a in config) / 11
        
        for x, y, angle in config:
            # Move toward center
            dx = center_x - x
            dy = center_y - y
            new_x = x + dx * (1 - scale_factor)
            new_y = y + dy * (1 - scale_factor)
            new_config.append((new_x, new_y, angle))
        
        # Resolve overlaps
        new_config = resolve_all_overlaps(new_config)
        
        # Check if better
        if is_configuration_valid(new_config):
            new_side = find_minimal_side(new_config)
            if new_side < best_side:
                best_config = new_config
                best_side = new_side
    
    return best_config, best_side


def generate_simple_packing():
    """Generate a simple but valid packing."""
    # Create a known valid configuration
    config = []
    
    # Center hexagon
    config.append((0.0, 0.0, 0.0))
    
    # First ring
    for i in range(6):
        angle = 60.0 * i
        x = SQRT3 * math.cos(math.radians(angle))
        y = SQRT3 * math.sin(math.radians(angle))
        config.append((x, y, 0.0))
    
    # Additional hexagons (carefully placed)
    extra_positions = [
        (2.0, 0.0, 30.0),
        (-2.0, 0.0, 30.0),
        (1.0, SQRT3, 0.0),
        (-1.0, -SQRT3, 0.0)
    ]
    
    for x, y, angle in extra_positions:
        config.append((x, y, angle))
    
    # Ensure no overlaps
    config = resolve_all_overlaps(config)
    
    side_length = find_minimal_side(config)
    
    return config, side_length


# EVOLVE-BLOCK-END


# Utility functions (unchanged)
# Note: The return result of optimize_construct needs to be verified by the verify_construction function 

def hexagon_vertices(center_x, center_y, side_length, angle_degrees):
    """
    Calculate the vertices of a hexagon.
    
    Args:
        center_x: x-coordinate of hexagon center
        center_y: y-coordinate of hexagon center
        side_length: Length of hexagon side
        angle_degrees: Rotation angle in degrees
        
    Returns:
        list: List of (x, y) tuples representing the hexagon vertices
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
    
    Args:
        v: (x, y) tuple representing the vector
        
    Returns:
        tuple: Normalized (x, y) vector
    """
    magnitude = math.sqrt(v[0]**2 + v[1]**2)
    return (v[0] / magnitude, v[1] / magnitude) if magnitude > EPSILON else (0., 0.)

def get_normals(vertices):
    """
    Calculate normal vectors for all edges of a polygon.
    
    Args:
        vertices: List of (x, y) tuples representing polygon vertices
        
    Returns:
        list: List of (x, y) normal vectors
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
    
    Args:
        vertices: List of (x, y) tuples representing polygon vertices
        axis: (x, y) tuple representing the projection axis
        
    Returns:
        tuple: (min_projection, max_projection) projection range
    """
    min_proj = float('inf')
    max_proj = float('-inf')
    for vertex in vertices:
        projection = vertex[0] * axis[0] + vertex[1] * axis[1]
        min_proj = min(min_proj, projection)
        max_proj = max(max_proj, projection)
    return min_proj, max_proj

def overlap_1d(min1, max1, min2, max2):
    """
    Check if two 1D intervals overlap (with epsilon tolerance).
    
    Args:
        min1, max1: Start and end of first interval
        min2, max2: Start and end of second interval
        
    Returns:
        bool: True if intervals overlap, False otherwise
    """
    return max1 >= min2 - EPSILON and max2 >= min1 - EPSILON

def polygons_intersect(vertices1, vertices2):
    """
    Check if two convex polygons intersect using the Separating Axis Theorem.
    
    Args:
        vertices1: List of (x, y) tuples for first polygon
        vertices2: List of (x, y) tuples for second polygon
        
    Returns:
        bool: True if polygons intersect, False otherwise
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
    
    Args:
        hex1_params: (center_x, center_y, side_length, angle_degrees) for first hexagon
        hex2_params: (center_x, center_y, side_length, angle_degrees) for second hexagon
        
    Returns:
        bool: True if hexagons are disjoint, False if they intersect
    """
    hex1_vertices = hexagon_vertices(*hex1_params)
    hex2_vertices = hexagon_vertices(*hex2_params)
    return not polygons_intersect(hex1_vertices, hex2_vertices)

def is_inside_hexagon(point, hex_params):
    """
    Check if a point is inside a hexagon.
    
    Args:
        point: (x, y) tuple representing the point
        hex_params: (center_x, center_y, side_length, angle_degrees) for the hexagon
        
    Returns:
        bool: True if point is inside hexagon, False otherwise
    """
    hex_vertices = hexagon_vertices(*hex_params)
    for i in range(len(hex_vertices)):
        p1 = hex_vertices[i]
        p2 = hex_vertices[(i + 1) % len(hex_vertices)]
        edge_vector = (p2[0] - p1[0], p2[1] - p1[1])
        point_vector = (point[0] - p1[0], point[1] - p1[1])
        cross_product = (edge_vector[0] * point_vector[1] - edge_vector[1] * point_vector[0])

        if cross_product < -EPSILON:
            return False

    return True

def all_hexagons_contained(inner_hex_params_list, outer_hex_params):
    """
    Check if all inner hexagons are completely contained within the outer hexagon.
    
    Args:
        inner_hex_params_list: List of (x, y, side_length, angle_degrees) for inner hexagons
        outer_hex_params: (center_x, center_y, side_length, angle_degrees) for outer hexagon
        
    Returns:
        bool: True if all inner hexagons are contained, False otherwise
    """
    for inner_hex_params in inner_hex_params_list:
        inner_hex_vertices = hexagon_vertices(*inner_hex_params)
        for vertex in inner_hex_vertices:
            if not is_inside_hexagon(vertex, outer_hex_params):
                return False
    return True

def verify_construction(inner_hex_data, outer_hex_center, outer_hex_side_length, outer_hex_angle_degrees):
    """
    Verify the complete construction meets all requirements.
    
    Args:
        inner_hex_data: List of [x, y, angle_degrees] for inner hexagons
        outer_hex_center: [x, y] for outer hexagon center
        outer_hex_side_length: Side length of outer hexagon
        outer_hex_angle_degrees: Rotation angle of outer hexagon in degrees
        
    Returns:
        bool: True if construction is valid, False otherwise
    """
    inner_hex_params_list = [
        (x, y, 1, angle) for x, y, angle in inner_hex_data
    ]
    outer_hex_params = (
        outer_hex_center[0], outer_hex_center[1],
        outer_hex_side_length, outer_hex_angle_degrees
    )

    if outer_hex_side_length < EPSILON:
        return False

    for i in range(len(inner_hex_params_list)):
        for j in range(i + 1, len(inner_hex_params_list)):
            if not hexagons_are_disjoint(inner_hex_params_list[i], inner_hex_params_list[j]):
                return False

    if not all_hexagons_contained(inner_hex_params_list, outer_hex_params):
        return False

    return True


if __name__ == "__main__":
    inner_hex_data, outer_hex, outer_side_length, outer_angle = optimize_construct()
    print(f"outer_side_length: {outer_side_length:.16f}")
    print(f"outer_angle: {outer_angle:.16f}")
    
    valid = verify_construction(inner_hex_data, outer_hex, outer_side_length, outer_angle)
    print(f"valid: {valid}")
    print(f"Target achieved: {outer_side_length < 3.931}")
