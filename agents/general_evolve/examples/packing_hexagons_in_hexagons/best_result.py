"""
Optimal hexagonal packing solution implementation.

This module provides functions for validating and visualizing hexagonal packing constructions.
It includes geometric calculations for hexagons, collision detection algorithms,
and visualization tools using matplotlib.

Key Features:
- Hexagon vertex generation with rotation
- Polygon intersection detection using Separating Axis Theorem
- Hexagon containment verification
- Visualization of packing configurations
"""

import numpy as np
import math
import matplotlib.pyplot as plt


def hexagon_vertices(
    center_x: float,
    center_y: float,
    side_length: float,
    angle_degrees: float,
) -> list[tuple[float, float]]:
    """Calculate vertices of a regular hexagon with given parameters.

    Args:
        center_x (float): X-coordinate of the hexagon center
        center_y (float): Y-coordinate of the hexagon center
        side_length (float): Length of each side of the hexagon
        angle_degrees (float): Rotation angle in degrees (clockwise from horizontal)

    Returns:
        list[tuple[float, float]]: List of (x,y) coordinate tuples for each vertex

    Note:
        The hexagon is always regular (all sides equal) and convex.
        Vertices are ordered clockwise starting from the rightmost point.
    """
    vertices = []
    angle_radians = math.radians(angle_degrees)
    for i in range(6):
        angle = angle_radians + 2 * math.pi * i / 6
        x = center_x + side_length * math.cos(angle)
        y = center_y + side_length * math.sin(angle)
        vertices.append((x, y))
    return vertices


def normalize_vector(v: tuple[float, float]) -> tuple[float, float]:
    """Normalize a 2D vector to unit length while preserving direction.

    Args:
        v (tuple[float, float]): Input vector as (x,y) tuple

    Returns:
        tuple[float, float]: Normalized vector with magnitude 1
        Returns (0,0) if input is zero vector
    """
    magnitude = math.sqrt(v[0] ** 2 + v[1] ** 2)
    return (v[0] / magnitude, v[1] / magnitude) if magnitude != 0 else (0.0, 0.0)


def get_normals(vertices: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Calculate outward-facing normal vectors for each edge of a polygon.

    Args:
        vertices (list[tuple[float, float]]): Polygon vertices in order

    Returns:
        list[tuple[float, float]]: Unit normal vectors for each edge

    Note:
        Normals are calculated by rotating each edge 90 degrees counter-clockwise
        and normalizing the resulting vector.
    """
    normals = []
    for i in range(len(vertices)):
        p1 = vertices[i]
        p2 = vertices[(i + 1) % len(vertices)]  # Wrap around to the first vertex.
        edge = (p2[0] - p1[0], p2[1] - p1[1])
        normal = normalize_vector((-edge[1], edge[0]))  # Rotate edge by 90 degrees.
        normals.append(normal)
    return normals


def project_polygon(
    vertices: list[tuple[float, float]],
    axis: tuple[float, float],
) -> tuple[float, float]:
    """Project polygon vertices onto an axis and return min/max projection values.

    Args:
        vertices (list[tuple[float, float]]): Polygon vertices
        axis (tuple[float, float]): Unit vector representing projection axis

    Returns:
        tuple[float, float]: (minimum projection value, maximum projection value)

    Note:
        Used in Separating Axis Theorem for collision detection
    """
    min_proj = float("inf")
    max_proj = float("-inf")
    for vertex in vertices:
        projection = vertex[0] * axis[0] + vertex[1] * axis[1]  # Dot product.
        min_proj = min(min_proj, projection)
        max_proj = max(max_proj, projection)
    return min_proj, max_proj


def overlap_1d(min1: float, max1: float, min2: float, max2: float) -> bool:
    """Check if two 1D intervals overlap.

    Args:
        min1 (float): Start of first interval
        max1 (float): End of first interval
        min2 (float): Start of second interval
        max2 (float): End of second interval

    Returns:
        bool: True if intervals overlap, False otherwise
    """
    return max1 >= min2 and max2 >= min1


def polygons_intersect(
    vertices1: list[tuple[float, float]],
    vertices2: list[tuple[float, float]],
) -> bool:
    """Check if two convex polygons intersect using Separating Axis Theorem.

    Args:
        vertices1 (list[tuple[float, float]]): Vertices of first polygon
        vertices2 (list[tuple[float, float]]): Vertices of second polygon

    Returns:
        bool: True if polygons intersect, False otherwise

    Algorithm:
        For each edge normal of both polygons, project both polygons onto the normal.
        If any projection shows no overlap, the polygons don't intersect.
    """
    normals1 = get_normals(vertices1)
    normals2 = get_normals(vertices2)
    axes = normals1 + normals2

    for axis in axes:
        min1, max1 = project_polygon(vertices1, axis)
        min2, max2 = project_polygon(vertices2, axis)
        if not overlap_1d(min1, max1, min2, max2):
            return False  # Separating axis found, polygons are disjoint.
    return True  # No separating axis found, polygons intersect.


def hexagons_are_disjoint(
    hex1_params: tuple[float, float, float, float],
    hex2_params: tuple[float, float, float, float],
) -> bool:
    """Check if two hexagons do not intersect.

    Args:
        hex1_params (tuple): (center_x, center_y, side_length, angle_degrees)
        hex2_params (tuple): (center_x, center_y, side_length, angle_degrees)

    Returns:
        bool: True if hexagons are disjoint, False if they overlap
    """
    hex1_vertices = hexagon_vertices(*hex1_params)
    hex2_vertices = hexagon_vertices(*hex2_params)
    return not polygons_intersect(hex1_vertices, hex2_vertices)


def is_inside_hexagon(
    point: tuple[float, float],
    hex_params: tuple[float, float, float, float],
) -> bool:
    """Determine if a point lies inside a hexagon.

    Args:
        point (tuple[float, float]): (x,y) coordinates of point to test
        hex_params (tuple): (center_x, center_y, side_length, angle_degrees)

    Returns:
        bool: True if point is inside hexagon, False otherwise

    Algorithm:
        Uses cross product to check if point is on same side of all edges
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
        if cross_product < 0:
            return False
    return True


def all_hexagons_contained(
    inner_hex_params_list: list[tuple[float, float, float, float]],
    outer_hex_params: tuple[float, float, float, float],
) -> bool:
    """Verify all inner hexagons are completely contained within outer hexagon.

    Args:
        inner_hex_params_list: List of inner hexagon parameters
        outer_hex_params: Parameters of containing hexagon

    Returns:
        bool: True if all inner hexagons are fully contained, False otherwise
    """
    for inner_hex_params in inner_hex_params_list:
        inner_hex_vertices = hexagon_vertices(*inner_hex_params)
        for vertex in inner_hex_vertices:
            if not is_inside_hexagon(vertex, outer_hex_params):
                return False
    return True


def verify_construction(
    inner_hex_data: tuple[float, float, float],
    outer_hex_center: tuple[float, float],
    outer_hex_side_length: float,
    outer_hex_angle_degrees: float,
):
    """Validate a hexagonal packing configuration.

    Args:
        inner_hex_data: List of (x, y, angle_degrees) for inner hexagons
        outer_hex_center: (x,y) center coordinates of outer hexagon
        outer_hex_side_length: Side length of outer hexagon
        outer_hex_angle_degrees: Rotation angle of outer hexagon in degrees

    Raises:
        AssertionError: If construction is invalid (overlaps or containment fails)

    Note:
        Inner hexagons are assumed to have side length 1
    """

    inner_hex_params_list = [
        (x, y, 1, angle) for x, y, angle in inner_hex_data
    ]  # Sets the side length to 1.
    outer_hex_params = (
        outer_hex_center[0],
        outer_hex_center[1],
        outer_hex_side_length,
        outer_hex_angle_degrees,
    )

    # Disjointness check.
    for i in range(len(inner_hex_params_list)):
        for j in range(i + 1, len(inner_hex_params_list)):
            if not hexagons_are_disjoint(
                inner_hex_params_list[i], inner_hex_params_list[j]
            ):
                raise AssertionError(f"Hexagons {i+1} and {j+1} intersect!")

    # Containment check.
    if not all_hexagons_contained(inner_hex_params_list, outer_hex_params):
        raise AssertionError(
            "Not all inner hexagons are contained in the outer hexagon!"
        )

    print("Construction is valid.")


def plot_construction(
    inner_hex_data: list[tuple[float, float, float]],
    outer_hex_center: tuple[float, float],
    outer_hex_side_length: float,
    outer_hex_angle_degrees: float,
):
    """Visualize hexagonal packing configuration using matplotlib.

    Args:
        inner_hex_data: List of (x, y, angle_degrees) for inner hexagons
        outer_hex_center: (x,y) center coordinates of outer hexagon
        outer_hex_side_length: Side length of outer hexagon
        outer_hex_angle_degrees: Rotation angle of outer hexagon in degrees

    Note:
        Outer hexagon is drawn in red, inner hexagons in semi-transparent blue
    """
    inner_hex_params_list = [(x, y, 1, angle) for x, y, angle in inner_hex_data]
    outer_hex_params = (
        outer_hex_center[0],
        outer_hex_center[1],
        outer_hex_side_length,
        outer_hex_angle_degrees,
    )

    _, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect("equal")

    # Plot outer hexagon (in red).
    outer_hex_vertices = hexagon_vertices(*outer_hex_params)
    outer_hex_x, outer_hex_y = zip(*outer_hex_vertices)
    ax.plot(
        outer_hex_x + (outer_hex_x[0],),
        outer_hex_y + (outer_hex_y[0],),
        "r-",
        label="Outer Hexagon",
    )

    # Plot inner hexagons (in blue).
    for i, inner_hex_params in enumerate(inner_hex_params_list):
        inner_hex_vertices = hexagon_vertices(*inner_hex_params)
        inner_hex_x, inner_hex_y = zip(*inner_hex_vertices)
        ax.fill(
            inner_hex_x + (inner_hex_x[0],),
            inner_hex_y + (inner_hex_y[0],),
            alpha=0.7,
            color="blue",
            label="Inner Hexagons" if i == 0 else None,  # Label only once.
        )

    ax.set_title("Hexagon Packing Construction")
    ax.legend()
    ax.grid(False)
    plt.show()


# Configuration of inner hexagons: [x, y, rotation_angle] for each hexagon
inner_hex_data = [
    [10.7365407326690470, 10.0651010025685217, 210.0348427040694048],
    [11.3871730701060017, 12.4023378805847830, 90.0215914575653215],
    [11.3784219638989139, 7.4636498925043178, 60.0180533439679138],
    [9.8539859496252884, 8.5554210038992924, 330.0690810530101089],
    [8.1220038949497138, 8.4477179958395467, 30.0622981937452387],
    [7.2553823156406185, 9.9487899411408698, 209.9216775103839154],
    [8.0282366951935522, 11.6576473762452899, 59.9948601778653483],
    [9.5210483575596001, 12.5363806520172520, 300.0097514724672010],
    [12.4214085189550190, 9.1212378850880036, 239.9856278441678796],
    [12.4361325723486615, 10.8533119273588543, 359.9886612146191851],
    [8.9880524744328660, 10.0563583299210642, 209.9816223890742322],
]

print("Number of inner hexagons:", len(inner_hex_data))

# Outer hexagon configuration
outer_hex_center = [
    np.float64(10.0000000000000000),
    np.float64(10.0000000000000000),
]  # Center coordinates
outer_hex_side_length = 3.9289068554637119  # Side length
outer_hex_angle_degrees = np.float64(0.0000000000000000)  # Rotation angle


# Validate the packing configuration
verify_construction(
    inner_hex_data, outer_hex_center, outer_hex_side_length, outer_hex_angle_degrees
)

# Print configuration details
print(f"Outer hexagon side length: {outer_hex_side_length}.")

# Visualize the packing
plot_construction(
    inner_hex_data, outer_hex_center, outer_hex_side_length, outer_hex_angle_degrees
)
