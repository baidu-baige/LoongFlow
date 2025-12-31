"""
Evaluator for circle packing example (n=26) with improved timeout handling
"""

import importlib.util
import os
import sys
import time
import traceback

import numpy as np

# subprocess is no longer used but kept/commented for import compatibility
# import subprocess
# import tempfile

num_circles = 26


class TimeoutError(Exception):
    """Custom exception raised when a function times out"""

    pass


def timeout_handler(signum, frame):
    """Handle timeout signal"""
    raise TimeoutError("Function execution timed out")


def _circles_overlap(centers, radii):
    """Protected function to compute max radii."""
    n = centers.shape[0]

    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
            if radii[i] + radii[j] > dist:
                return True

    return False


def check_construction(centers, radii, n):
    """Evaluates circle packing for maximizing sum of radii in unit square."""

    # General checks for the whole array
    if centers.shape != (n, 2):
        return {
            "sum_of_radii": -np.inf
        }, f"Error: The 'centers' array has an invalid shape"

    if not np.isfinite(centers).all():
        return {
            "sum_of_radii": -np.inf
        }, f"Error: The 'centers' array has non-finite values."

    # --- Start of the modified geometric check ---

    # 1. Check each circle individually to see if it's contained
    is_contained = ((radii[:, None] <= centers) & (centers <= 1 - radii[:, None])).all(
        axis=1
    )

    # 2. If not all of them are contained...
    if not is_contained.all():
        return {"sum_of_radii": -np.inf}, "Error: Not all radii are contained"

    if radii.shape != (n,) or not np.isfinite(radii).all() or not (0 <= radii).all():
        return {
            "sum_of_radii": -np.inf
        }, f"Error: radii shape {radii.shape} != ({n} or existed radii < 0)"

    if _circles_overlap(centers, radii):
        return {"sum_of_radii": -np.inf}, "Error: circles overlap"

    print("The circles are disjoint and lie inside the unit square.")
    return {"sum_of_radii": float(np.sum(radii))}, ""


def validate_packing(centers, radii):
    """
    Validate that circles don't overlap and are inside the unit square
    **WITHOUT numerical tolerance**.
    """
    n = centers.shape[0]

    if np.isnan(centers).any() or np.isnan(radii).any():
        print("NaN values detected in solution", file=sys.stderr)
        return False

    if np.any(radii < 0):
        print(f"Detected negative radii", file=sys.stderr)
        return False

    # Check if circles are inside the unit square (strict check)
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        if x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
            print(
                f"Circle {i} at ({x:.8f}, {y:.8f}) with radius {r:.18f} is outside the unit square",
                file=sys.stderr,
            )
            return False

    # Check for overlaps (strict check)
    for i in range(n):
        for j in range(i + 1, n):
            sq_dist = np.sum((centers[i] - centers[j]) ** 2)
            sq_sum_radii = (radii[i] + radii[j]) ** 2

            # Use a tiny machine-epsilon-level tolerance for the comparison itself
            # This handles cases where sq_dist and sq_sum_radii are truly identical in theory
            # but differ by the smallest possible float amount.
            if sq_dist < sq_sum_radii - 1e-30:
                dist = np.sqrt(sq_dist)
                print(
                    f"Circles {i} and {j} overlap: dist={dist:.18f}, r1+r2={radii[i]+radii[j]:.18f}",
                    file=sys.stderr,
                )
                return False

    return True


def run_with_timeout(program_path, timeout_seconds=20):
    """
    Run the program using its existing unique filename.
    Supports multiprocessing naturally.
    """
    print(f"Executing program: {program_path}")

    program_dir, file_name = os.path.split(program_path)
    module_name, _ = os.path.splitext(file_name)

    if not module_name.isidentifier():
        raise ValueError(
            f"Invalid module name: '{module_name}'. "
            "Filename must contain only letters, numbers, and underscores, "
            "and cannot start with a number."
        )

    if program_dir not in sys.path:
        sys.path.insert(0, program_dir)

    try:
        if module_name in sys.modules:
            program_module = importlib.reload(sys.modules[module_name])
        else:
            program_module = importlib.import_module(module_name)

        if not hasattr(program_module, "run_packing"):
            raise AttributeError(f"Function 'run_packing' not found in {program_path}")

        print(f"Calling run_packing(num_circles={num_circles})...")

        centers, radii, sum_radii = program_module.run_packing(num_circles=num_circles)

        print(f"run_packing() returned successfully: sum_radii = {sum_radii}")
        return centers, radii, sum_radii

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

    The returned dictionary MUST contain a 'status' key with one of the following values:
    - "execution_failed": The program code failed to run (e.g., crash, timeout).
    - "validation_failed": The program ran, but its output was invalid.
    - "success": The program ran and produced a valid, scored result.
    """
    TARGET_VALUE = 2.6358627564136983  # AlphaEvolve result for n=26
    start_time = time.time()

    try:
        centers, radii, reported_sum = run_with_timeout(
            program_path, timeout_seconds=3600
        )
        eval_time = time.time() - start_time

        if not isinstance(centers, np.ndarray):
            centers = np.array(centers)
        if not isinstance(radii, np.ndarray):
            radii = np.array(radii)

    except Exception as e:
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

    if (
        centers.shape != (26, 2)
        or radii.shape != (26,)
        or np.isnan(centers).any()
        or np.isnan(radii).any()
    ):
        summary = (
            f"Validation failed: Output shapes are incorrect or contain NaN."
            f"Centers: {centers.shape}, Radii: {radii.shape}. "
        )
        print(summary, file=sys.stderr)
        return {
            "status": "validation_failed",
            "summary": summary,
            "score": 0.0,
            "metrics": {"validity": 0.0},
            "artifacts": {"reason": "Invalid shape or NaN values"},
        }

    validation_result, error_msg = check_construction(centers, radii, 26)
    if validation_result["sum_of_radii"] == -np.inf:
        summary = f"Validation failed: {error_msg}"
        print(summary, file=sys.stderr)
        return {
            "status": "validation_failed",
            "summary": summary,
            "score": 0.0,
            "metrics": {"validity": 0.0, "sum_radii": 0.0},
            "artifacts": {"reason": "Geometric constraints not met."},
        }

    sum_radii = np.sum(radii)
    target_ratio = sum_radii / TARGET_VALUE

    score = target_ratio

    summary = f"Success: Valid packing found. Sum of radii: {sum_radii:.6f}, Score: {score:.4f}"
    print(summary)

    return {
        "status": "success",
        "summary": summary,
        "score": float(score),
        "metrics": {
            "sum_radii": float(sum_radii),
            "target_ratio": float(target_ratio),
            "validity": 1.0,
            "eval_time": float(eval_time),
        },
    }
