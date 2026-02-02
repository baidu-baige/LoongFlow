"""
Evaluator for the Heilbronn problem for convex regions (n=13), aligned with standard specifications.
"""

import os
import pickle
import subprocess
import sys
import tempfile
import time
import traceback

import numpy as np
from scipy.spatial import ConvexHull

# --- Constants ---
N_POINTS = 13
TARGET_VALUE = 0.0309  # Benchmark for n=13 in a unit square
TIMEOUT_SECONDS = 1800
TOL = 1e-6


class TimeoutError(Exception):
    """Custom timeout exception."""

    pass


def validate_placement(points: np.ndarray):
    """Checks that all points are inside the unit square [0,1]x[0,1],
    that no points overlap, and that any three points can form a triangle."""

    def is_collinear(p1, p2, p3):
        # Using a tolerance for floating point comparisons
        return (
            abs(
                p1[0] * (p2[1] - p3[1])
                + p2[0] * (p3[1] - p1[1])
                + p3[0] * (p1[1] - p2[1])
            )
            < TOL
        )

    # Check for duplicates by checking distance between all pairs of points
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            if np.linalg.norm(points[i] - points[j]) < TOL:
                return False, f"Duplicate points found: P{i} and P{j} are too close."

    # Check if points are inside the unit square
    for i, (x, y) in enumerate(points):
        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
            return False, f"Point P{i} ({x:.4f}, {y:.4f}) is outside the unit square."

    # Check if any three points are collinear
    num_points = len(points)
    for i in range(num_points):
        for j in range(i + 1, num_points):
            for k in range(j + 1, num_points):
                if is_collinear(points[i], points[j], points[k]):
                    return False, f"Points P{i}, P{j}, P{k} are collinear."

    return True, None


def triangle_area(a, b, c):
    """Calculate the area of a triangle given three vertices using the cross product formula."""
    return 0.5 * abs(a[0] * (b[1] - c[1]) + b[0] * (c[1] - a[1]) + c[0] * (a[1] - b[1]))


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


def run_with_timeout(program_path, n_points, timeout_seconds=TIMEOUT_SECONDS):
    """
    Runs the program in a separate process with a timeout.
    """
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as temp_file:
        script = f"""
import sys
import numpy as np
import os
import pickle
import traceback

sys.path.insert(0, os.path.dirname('{program_path}'))

try:
    spec = __import__('importlib.util').util.spec_from_file_location("program", '{program_path}')
    program = __import__('importlib.util').util.module_from_spec(spec)
    spec.loader.exec_module(program)

    print(f"Calling run_search_point(n={n_points})...")
    points, min_area = program.run_search_point(n={n_points})
    print(f"run_search_point() returned successfully.")

    results = {{'points': points, 'min_area': min_area}}
    with open('{temp_file.name}.results', 'wb') as f:
        pickle.dump(results, f)

except Exception as e:
    with open('{temp_file.name}.results', 'wb') as f:
        pickle.dump({{'error': str(e), 'traceback': traceback.format_exc()}}, f)
"""
        temp_file.write(script.encode())
        temp_file_path = temp_file.name

    results_path = f"{temp_file_path}.results"
    try:
        process = subprocess.Popen(
            [sys.executable, temp_file_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        stdout, stderr = process.communicate(timeout=timeout_seconds)

        if process.returncode != 0:
            raise RuntimeError(
                f"Process exited with code {process.returncode}\\n"
                f"stdout:\\n{stdout.decode()}\\nstderr:\\n{stderr.decode()}"
            )

        if os.path.exists(results_path):
            with open(results_path, "rb") as f:
                results = pickle.load(f)
            if "error" in results:
                raise RuntimeError(
                    f"Program execution failed: {results['error']}\\n{results['traceback']}"
                )
            return results["points"], results["min_area"]
        else:
            raise RuntimeError("Results file not found.")
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
        raise TimeoutError(f"Process timed out after {timeout_seconds} seconds")
    finally:
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)
        if os.path.exists(results_path):
            os.unlink(results_path)


def evaluate(program_path):
    """
    Evaluates the program and returns a structured result dictionary.
    """
    start_time = time.time()
    status = "success"

    try:
        points, reported_min_area = run_with_timeout(program_path, n_points=N_POINTS)
        eval_time = time.time() - start_time

        if not isinstance(points, np.ndarray):
            points = np.array(points)

        # --- Shape Validation ---
        if points.shape != (N_POINTS, 2):
            shape_error = (
                f"Invalid shape: expected ({N_POINTS}, 2), but got {points.shape}"
            )
            return {
                "score": 0.0,
                "status": "validation_failed",
                "summary": shape_error,
                "metrics": {
                    "min_area_ratio": 0.0,
                    "target_ratio": 0.0,
                    "validity": 0.0,
                    "eval_time": eval_time,
                },
                "artifacts": {
                    "stderr": shape_error,
                    "failure_stage": "shape_validation",
                    "execution_time": f"{eval_time:.2f}s",
                },
            }

        # --- Geometric Validation ---
        is_valid, error_message = validate_placement(points)
        validity = 1.0 if is_valid else 0.0

        # --- Independent Metric Calculation ---
        final_min_area = min_triangle_area(points) if is_valid else 0.0

        # Calculate area ratio based on the convex hull of the points
        min_area_ratio = 0.0
        if is_valid:
            try:
                hull = ConvexHull(points)
                convex_hull_area = hull.volume
                if convex_hull_area > TOL:
                    min_area_ratio = final_min_area / convex_hull_area
            except Exception as e:
                is_valid = False
                validity = 0.0
                error_message = f"Failed to compute convex hull: {e}"

        target_ratio = min_area_ratio / TARGET_VALUE if is_valid else 0.0

        # The final score is the target ratio, penalized if invalid
        score = target_ratio * validity

        artifacts = {"execution_time": f"{eval_time:.2f}s"}
        if not is_valid:
            status = "validation_failed"
            summary = f"Validation failed: {error_message}"
            artifacts["validation_report"] = f"Validation failed: {error_message}"
            artifacts["failure_stage"] = "geometric_validation"
        else:
            status = "success"
            summary = "Evaluation successful."
            artifacts["validation_report"] = "Placement is valid."
            artifacts["summary"] = f"Achieved {target_ratio:.2%} of benchmark."

        return {
            "score": float(score),
            "status": status,
            "summary": summary,
            "metrics": {
                "min_area": float(final_min_area),
                "min_area_ratio": float(min_area_ratio),
                "target_ratio": float(target_ratio),
                "validity": float(validity),
                "eval_time": float(eval_time),
            },
            "artifacts": artifacts,
        }

    except TimeoutError as e:
        eval_time = time.time() - start_time
        return {
            "score": 0.0,
            "status": "execution_failed",
            "summary": f"Execution failed: The program timed out after {TIMEOUT_SECONDS} seconds.",
            "metrics": {
                "min_area_ratio": 0.0,
                "target_ratio": 0.0,
                "validity": 0.0,
                "eval_time": eval_time,
            },
            "artifacts": {
                "stderr": str(e),
                "failure_stage": "execution_timeout",
                "execution_time": f"{eval_time:.2f}s",
            },
        }
    except Exception as e:
        eval_time = time.time() - start_time
        return {
            "score": 0.0,
            "status": "execution_failed",
            "summary": f"Program execution failed: {str(e)}",
            "metrics": {
                "min_area_ratio": 0.0,
                "target_ratio": 0.0,
                "validity": 0.0,
                "eval_time": eval_time,
            },
            "artifacts": {
                "stderr": f"Evaluation failed completely: {str(e)}",
                "traceback": traceback.format_exc(),
                "failure_stage": "program_execution",
                "execution_time": f"{eval_time:.2f}s",
            },
        }


if __name__ == "__main__":
    file = "initial_program.py"
    res = evaluate(file)
    import json

    print(json.dumps(res, ensure_ascii=False, indent=4))
