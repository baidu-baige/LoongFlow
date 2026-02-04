"""
Evaluator for Hexagon packing example (n=11)
Modified to support construct_packing interface and hexagon geometry validation.
"""

import json
import importlib.util
import os
import sys
import time
import traceback
import math
import numpy as np

# Define a global floating-point tolerance for comparisons.
EPSILON = 1e-9


def hexagon_vertices(
    center_x: float,
    center_y: float,
    side_length: float,
    angle_degrees: float,
) -> list[tuple[float, float]]:
    """Generates the vertices of a regular hexagon.

    Args:
        center_x: x-coordinate of the center.
        center_y: y-coordinate of the center.
        side_length: Length of each side.
        angle_degrees: Rotation angle in degrees (clockwise from horizontal).

    Returns:
        A list of tuples, where each tuple (x, y) represents the vertex location.
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
    """Normalizes a 2D vector."""
    magnitude = math.sqrt(v[0] ** 2 + v[1] ** 2)
    # Define a small tolerance to check for near-zero magnitude
    return (v[0] / magnitude, v[1] / magnitude) if magnitude > EPSILON else (0.0, 0.0)


def get_normals(vertices: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Gets the outward normals of a polygon's edges."""
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
    """Projects a polygon onto an axis and returns the min/max values."""
    min_proj = float("inf")
    max_proj = float("-inf")
    for vertex in vertices:
        projection = vertex[0] * axis[0] + vertex[1] * axis[1]  # Dot product.
        min_proj = min(min_proj, projection)
        max_proj = max(max_proj, projection)
    return min_proj, max_proj


# Chekk if two 1D intervals overlap
def overlap_1d(min1: float, max1: float, min2: float, max2: float) -> bool:
    """Determines whether two 1D intervals overlap."""
    # The condition for non-overlap is: max1 < min2 or max2 < min1
    # Consider floating-point precision, use EPSILON tolerance.
    return max1 >= min2 - EPSILON and max2 >= min1 - EPSILON


def polygons_intersect(
    vertices1: list[tuple[float, float]],
    vertices2: list[tuple[float, float]],
) -> bool:
    """Determines if two polygons intersect using the Separating Axis Theorem."""
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
    """Determines if two hexagons are disjoint given their parameters."""
    hex1_vertices = hexagon_vertices(*hex1_params)
    hex2_vertices = hexagon_vertices(*hex2_params)
    return not polygons_intersect(hex1_vertices, hex2_vertices)


# Check if a point is inside a hexagon with tolerance
def is_inside_hexagon(
    point: tuple[float, float],
    hex_params: tuple[float, float, float, float],
) -> bool:
    """Checks if a point is inside a hexagon (given its parameters)."""
    hex_vertices = hexagon_vertices(*hex_params)
    for i in range(len(hex_vertices)):
        p1 = hex_vertices[i]
        p2 = hex_vertices[(i + 1) % len(hex_vertices)]
        edge_vector = (p2[0] - p1[0], p2[1] - p1[1])
        point_vector = (point[0] - p1[0], point[1] - p1[1])
        cross_product = (
            edge_vector[0] * point_vector[1] - edge_vector[1] * point_vector[0]
        )

        # Use tolerance check: if cross_product is significantly
        # less than 0 (i.e., less than -EPSILON), the point is outside.
        # If cross_product is between [-EPSILON, 0], consider the point on the boundary or inside.
        if cross_product < -EPSILON:
            return False

    return True


def all_hexagons_contained(
    inner_hex_params_list: list[tuple[float, float, float, float]],
    outer_hex_params: tuple[float, float, float, float],
) -> bool:
    """Checks if all inner hexagons are contained within the outer hexagon."""
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
    """Verifies the hexagon packing construction with a rotated outer hexagon.

    Args:
        inner_hex_data: List of (x, y, angle_degrees) tuples for inner hexagons.
        outer_hex_center: (x, y) tuple for the outer hexagon center.
        outer_hex_side_length: Side length of the outer hexagon.
        outer_hex_angle_degrees: Rotation angle of the outer hexagon in degrees.

    Raises:
        AssertionError if the construction is not valid.
    """

    validation_details = {
        "validation_result": [],
    }

    inner_hex_params_list = [
        (x, y, 1, angle) for x, y, angle in inner_hex_data
    ]  # Sets the side length to 1.
    outer_hex_params = (
        outer_hex_center[0],
        outer_hex_center[1],
        outer_hex_side_length,
        outer_hex_angle_degrees,
    )

    if len(inner_hex_data) != 11:
        msg = f"Validation failed: the number of internal hexagons is not 11!"
        validation_details["validation_result"].append(msg)
        print(msg)
        return False, validation_details

    # Side length check
    if outer_hex_side_length < 1e-10:
        msg = f"Validation failed: outer_hex_side_length can not is 0!"
        validation_details["validation_result"].append(msg)
        print(msg)
        return False, validation_details

    # Disjointness check.
    for i in range(len(inner_hex_params_list)):
        for j in range(i + 1, len(inner_hex_params_list)):
            if not hexagons_are_disjoint(
                inner_hex_params_list[i], inner_hex_params_list[j]
            ):
                msg = f"Validation failed: Hexagons {i} and {j} intersect!"
                validation_details["validation_result"].append(msg)
                print(msg)
                return False, validation_details

    # Containment check.
    if not all_hexagons_contained(inner_hex_params_list, outer_hex_params):
        msg = "Not all inner hexagons are contained in the outer hexagon!"
        validation_details["validation_result"].append(msg)
        print(msg)
        return False, validation_details

    msg = "Validation passed! All constraints satisfied."
    validation_details["validation_result"].append(msg)
    print(msg)
    return True, validation_details


class TimeoutError(Exception):
    """Custom exception raised when a function times out"""

    pass


def timeout_handler(signum, frame):
    """Handle timeout signal"""
    raise TimeoutError("Function execution timed out")


def run_with_timeout(program_path, timeout_seconds=20):
    """
    Run the program using its existing unique filename.
    Expected to contain 'construct_packing' function.
    """
    print(f"Executing program: {program_path}")

    program_dir, file_name = os.path.split(program_path)
    module_name, _ = os.path.splitext(file_name)

    if not module_name.isidentifier():
        raise ValueError(f"Invalid module name: '{module_name}'")

    if program_dir not in sys.path:
        sys.path.insert(0, program_dir)

    try:
        if module_name in sys.modules:
            program_module = importlib.reload(sys.modules[module_name])
        else:
            program_module = importlib.import_module(module_name)

        if not hasattr(program_module, "optimize_construct"):
            raise AttributeError(
                f"Function 'optimize_construct' not found in {program_path}"
            )

        print("Calling optimize_construct()...")

        # Updated call signature based on Code 2
        result = program_module.optimize_construct()

        # Validate return structure roughly
        if not isinstance(result, tuple) or len(result) != 4:
            raise ValueError("optimize_construct must return a tuple of length 4")

        (
            inner_hex_data,
            outer_hex_center,
            outer_hex_side_length,
            outer_hex_angle_degrees,
        ) = result

        print(
            f"optimize_construct() returned successfully. outer_hex_side_length: {outer_hex_side_length}"
        )
        return (
            inner_hex_data,
            outer_hex_center,
            outer_hex_side_length,
            outer_hex_angle_degrees,
        )

    except Exception as e:
        print(f"Error during execution: {str(e)}")
        raise RuntimeError(f"Program execution failed: {str(e)}") from e

    finally:
        if program_dir in sys.path:
            sys.path.remove(program_dir)
        if module_name in sys.modules:
            del sys.modules[module_name]


def evaluate(program_path):
    """
    Evaluates the program and returns a dictionary adhering to the specified contract.
    """
    start_time = time.time()
    TARGET = 3.931

    try:
        # 1. --- Execution Phase ---
        (
            inner_hex_data,
            outer_hex_center,
            outer_hex_side_length,
            outer_hex_angle_degrees,
        ) = run_with_timeout(program_path, timeout_seconds=1800)
        eval_time = time.time() - start_time

    except Exception as e:
        # --- Execution Failed ---
        error_msg = f"Program execution failed: {str(e)}"
        print(error_msg, file=sys.stderr)
        traceback.print_exc()
        return {
            "status": "execution_failed",
            "score": 0.0,
            "summary": error_msg,
            "artifacts": {
                "error_type": type(e).__name__,
                "traceback": traceback.format_exc(),
            },
        }

    # 2. --- Validation Phase ---
    is_valid, validation_details = verify_construction(
        inner_hex_data, outer_hex_center, outer_hex_side_length, outer_hex_angle_degrees
    )

    if not is_valid:
        error_msg = json.dumps(validation_details, ensure_ascii=False, indent=2)
        summary = f"Validation failed: {error_msg}"
        print(summary, file=sys.stderr)
        return {
            "status": "validation_failed",
            "summary": summary,
            "score": 0.0,
            "metrics": {"validity": 0.0},
            "artifacts": {"reason": error_msg},
        }

    # 3. --- Success Phase ---

    score = TARGET / outer_hex_side_length

    summary = f"Success: Valid construction. Outer hex side length: {outer_hex_side_length:.16f}"
    print(summary)

    return {
        "status": "success",
        "summary": summary,
        "score": float(score),
        "metrics": {
            "outer_hex_side_length": float(outer_hex_side_length),
            "target_ratio": float(score),
            "validity": 1.0,
            "eval_time": float(eval_time),
        },
    }


if __name__ == "__main__":
    file = "./initial_program.py"
    evaluate(file)
