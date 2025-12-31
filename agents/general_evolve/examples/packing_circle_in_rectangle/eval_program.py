"""
Evaluator for circle packing example (n=21) with improved timeout handling
Enhanced with artifacts to demonstrate execution feedback
"""

import itertools
import os
import pickle
import subprocess
import sys
import tempfile
import time
import traceback

import numpy as np

num_circles = 21


class TimeoutError(Exception):
    """Custom timeout exception"""

    pass


def timeout_handler(signum, frame):
    """Handle timeout signal"""
    raise TimeoutError("Function execution timed out")


def minimum_circumscribing_rectangle(circles: np.ndarray) -> tuple[float, float]:
    """Returns the width and height of the minimum circumscribing rectangle.

    Args:
        circles: A numpy array of shape (num_circles, 3), where each row is of the
            form (x, y, radius), specifying a circle.

    Returns:
        A tuple (width, height) of the minimum circumscribing rectangle.
    """
    min_x = np.min(circles[:, 0] - circles[:, 2])
    max_x = np.max(circles[:, 0] + circles[:, 2])
    min_y = np.min(circles[:, 1] - circles[:, 2])
    max_y = np.max(circles[:, 1] + circles[:, 2])
    return max_x - min_x, max_y - min_y


def validate_packing(circles: np.ndarray):
    """
    Validate that circles don't overlap and are inside a rectangle of perimeter 4.

    Args:
        circles: A numpy array of shape (num_circles, 3), where each row is of the
         form (x, y, radius), specifying a circle.

    Returns:
        Tuple of (is_valid: bool, validation_details: dict)
    """

    num_circles = len(circles)

    validation_details = {
        "total_circles": num_circles,
        "overlaps_check": [],
        "perimeter_check": [],
        "min_radius": float(np.min(circles[:, 2])),
        "max_radius": float(np.max(circles[:, 2])),
        "avg_radius": float(np.mean(circles[:, 2])),
        "perimeter": 0.0,
    }

    # Checks that circles are disjoint
    for circle1, circle2 in itertools.combinations(circles, 2):
        center_distance = np.sqrt(
            (circle1[0] - circle2[0]) ** 2 + (circle1[1] - circle2[1]) ** 2
        )
        radii_sum = circle1[2] + circle2[2]
        if center_distance < radii_sum:
            violation = f"Circles are NOT disjoint: {circle1} and {circle2}."
            validation_details["overlaps_check"].append(violation)
            print(violation)

    # Checks rectangle of perimeter 4
    width, height = minimum_circumscribing_rectangle(circles)
    perimeter = 2 * (width + height)
    validation_details["perimeter"] = perimeter
    if (width + height) > 2:
        violation = f"Perimeter of minimum circumscribing rectangle: {perimeter:.6f}, not equal to 4"
        validation_details["perimeter_check"].append(violation)
        print(violation)

    is_valid = (
        len(validation_details["overlaps_check"]) == 0
        and len(validation_details["perimeter_check"]) == 0
    )

    validation_details["is_valid"] = is_valid

    return is_valid, validation_details


def _circles_overlap(centers, radii):
    """Protected function to compute max radii."""
    n = centers.shape[0]

    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
            if radii[i] + radii[j] > dist:
                return True

    return False


def check_construction_rectangle(
    centers: np.ndarray, radii: np.ndarray, n: int, width: float, height: float
) -> dict:
    """
    Evaluates a circle packing in a rectangle.

    Checks if all circles are contained within the rectangle and do not overlap.
    Provides detailed diagnostics for any violations, distinguishing between
    genuine errors and potential floating-point precision issues.

    Args:
      centers: A numpy array of shape (n, 2) with the (x, y) coordinates of the circle centers.
      radii: A numpy array of shape (n,) with the radii of the circles.
      n: The number of circles.
      width: The width of the rectangle.
      height: The height of the rectangle.

    Returns:
      A dictionary containing the sum of radii if the packing is valid.
      If invalid, it returns a dictionary with -np.inf as the sum of radii
      and a corresponding error_message.
    """

    TOLERANCE = 1e-9  # Tolerance for floating-point comparisons

    # --- Start of checks for rectangle geometry ---
    # 1. Check if width and height are finite, real numbers.
    if not np.all(np.isfinite([width, height])) or not np.isrealobj(
        np.array([width, height])
    ):
        error_message = "Invalid width or height. Must be finite real numbers."
        print(f"Error: {error_message}")
        return {"sum_of_radii": -np.inf, "error_message": error_message}

    # 2. Check if the rectangle's perimeter is 4.
    if not np.isclose(2 * (width + height), 4.0):
        error_message = f"Perimeter is not 4. Got {2 * (width + height)}"
        print(f"Error: {error_message}")
        return {"sum_of_radii": -np.inf, "error_message": error_message}

    # 3. Check for valid, non-degenerate rectangle dimensions.
    if width <= 0 or height <= 0:
        error_message = f"Invalid rectangle dimensions. width={width}, height={height}"
        print(f"Error: {error_message}")
        return {"sum_of_radii": -np.inf, "error_message": error_message}
    # --- End of rectangle checks ---

    # General checks for the input arrays
    if (
        centers.shape != (n, 2)
        or not np.isfinite(centers).all()
        or not np.isrealobj(centers)
    ):
        error_message = (
            "The 'centers' array has an invalid shape, non-finite, or complex values."
        )
        print(f"Error: {error_message}")
        return {"sum_of_radii": -np.inf, "error_message": error_message}

    # --- Geometric check for circle containment ---
    # 1. Check each circle individually to see if it's contained
    is_contained = (
        (radii[:, None] <= centers)
        & (centers <= np.array([width, height]) - radii[:, None])
    ).all(axis=1)

    # 2. If not all of them are contained, print diagnostics
    if not is_contained.all():
        error_message = "Circles are not contained within the rectangle."
        print(f"Error: {error_message}")
        for i, contained in enumerate(is_contained):
            if not contained:
                print(f"-> Diagnostics for Circle {i}:")
                c_i = centers[i]
                r_i = radii[i]
                # Check violation for each of the four boundaries
                violations = {
                    "left": r_i - c_i[0],
                    "right": c_i[0] - (width - r_i),
                    "bottom": r_i - c_i[1],
                    "top": c_i[1] - (height - r_i),
                }
                for boundary, violation_amount in violations.items():
                    if violation_amount > TOLERANCE:
                        print(
                            f"  - Genuinely violates {boundary} boundary by {violation_amount:.4g}"
                        )
                    elif violation_amount > 0:
                        print(
                            f"  - Potential precision error at {boundary} boundary. Violation: {violation_amount:.4g}"
                        )

        return {"sum_of_radii": -np.inf, "error_message": error_message}

    # --- Geometric check for circle overlaps ---
    if n > 1:
        has_overlap = False
        # Iterate over every unique pair of circles
        for i, j in itertools.combinations(range(n), 2):
            center_dist_sq = np.sum((centers[i] - centers[j]) ** 2)
            radii_sum_sq = (radii[i] + radii[j]) ** 2

            # Check if squared distance is less than squared sum of radii
            if center_dist_sq < radii_sum_sq:
                if not has_overlap:  # Print header only once
                    print("Error: Circles are overlapping.")
                    has_overlap = True

                overlap_sq = radii_sum_sq - center_dist_sq
                # Distinguish between genuine overlap and touching circles (precision issue)
                if overlap_sq > TOLERANCE:
                    print(
                        f"  - Genuinely overlapping: Circles {i} and {j}. Squared overlap: {overlap_sq:.4g}"
                    )
                else:
                    print(
                        f"  - Potential precision error: Circles {i} and {j} are touching/minutely overlapping. \
Squared overlap: {overlap_sq:.4g}"
                    )

        if has_overlap:
            error_message = "Circles are overlapping."
            return {"sum_of_radii": -np.inf, "error_message": error_message}

    if (
        radii.shape != (n,)
        or not np.isfinite(radii).all()
        or not (0 <= radii).all()
        or not np.isrealobj(radii)  # Added check for real numbers
    ):
        error_message = "radii bad shape or contains non-real/non-finite values"
        print(error_message)
        return {"sum_of_radii": -np.inf, "error_message": error_message}

    if _circles_overlap(centers, radii):
        error_message = "circles overlap"
        print(error_message)
        # Note: The original return value here was `({'sum_of_radii': -np.inf}, {})`, which was a tuple.
        # It has been corrected to a dictionary to be consistent with other failure cases.
        return {"sum_of_radii": -np.inf, "error_message": error_message}

    print(
        f"Valid packing found with width={width}, height={height},"
        f" sum_radii={np.sum(radii)}"
    )

    print("The circles are disjoint and lie inside the rectangle.")
    return {"sum_of_radii": float(np.sum(radii))}


def run_with_timeout(program_path, timeout_seconds=20):
    """
    Run the program in a separate process with timeout
    using a simple subprocess approach

    Args:
        program_path: Path to the program file
        timeout_seconds: Maximum execution time in seconds

    Returns:
        circles from the program
    """
    # Create a temporary file to execute
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as temp_file:
        # Write a script that executes the program and saves results
        script = f"""
import sys
import numpy as np
import os
import pickle
import traceback

# Add the directory to sys.path
sys.path.insert(0, os.path.dirname('{program_path}'))

# Debugging info
print(f"Running in subprocess, Python version: {{sys.version}}")
print(f"Program path: {program_path}")

try:
    # Import the program
    spec = __import__('importlib.util').util.spec_from_file_location("program", '{program_path}')
    program = __import__('importlib.util').util.module_from_spec(spec)
    spec.loader.exec_module(program)
    
    # Run the packing function
    print("Calling run_packing(num_circles={num_circles})...")
    centers, radii, width, height = program.run_packing(num_circles={num_circles})
    print(f"run_packing() returned successfully: centers = {{centers}}")

    # Save results to a file
    results = {{
        'centers': centers,
        'radii': radii,
        'width': width,
        'height': height,
    }}

    with open('{temp_file.name}.results', 'wb') as f:
        pickle.dump(results, f)
    print(f"Results saved to {temp_file.name}.results")
    
except Exception as e:
    # If an error occurs, save the error instead
    print(f"Error in subprocess: {{str(e)}}")
    traceback.print_exc()
    with open('{temp_file.name}.results', 'wb') as f:
        pickle.dump({{'error': str(e)}}, f)
    print(f"Error saved to {temp_file.name}.results")
"""
        temp_file.write(script.encode())
        temp_file_path = temp_file.name

    results_path = f"{temp_file_path}.results"

    try:
        # Run the script with timeout
        process = subprocess.Popen(
            [sys.executable, temp_file_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        try:
            stdout, stderr = process.communicate(timeout=timeout_seconds)
            exit_code = process.returncode

            # Always print output for debugging purposes
            print(f"Subprocess stdout: {stdout.decode()}")
            if stderr:
                print(f"Subprocess stderr: {stderr.decode()}")

            # Still raise an error for non-zero exit codes, but only after printing the output
            if exit_code != 0:
                raise RuntimeError(f"Process exited with code {exit_code}")

            # Load the results
            if os.path.exists(results_path):
                with open(results_path, "rb") as f:
                    results = pickle.load(f)

                # Check if an error was returned
                if "error" in results:
                    raise RuntimeError(f"Program execution failed: {results['error']}")

                return (
                    results["centers"],
                    results["radii"],
                    results["width"],
                    results["height"],
                )
            else:
                raise RuntimeError("Results file not found")

        except subprocess.TimeoutExpired:
            # Kill the process if it times out
            process.kill()
            process.wait()
            raise TimeoutError(f"Process timed out after {timeout_seconds} seconds")

    finally:
        # Clean up temporary files
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)
        if os.path.exists(results_path):
            os.unlink(results_path)


def evaluate(program_path):
    """
    Evaluate the program by running it once and checking the sum of radii.
    The returned dictionary conforms to the new specified structure.

    Args:
        program_path: Path to the program file

    Returns:
        A dictionary with 'status', 'summary', 'score', 'metrics', and 'artifacts'.
    """
    # Target value from the paper
    TARGET_VALUE = 2.365  # AlphaEvolve result for n=21
    timeout_duration = 3600
    status = "success"

    try:
        start_time = time.time()

        # Use subprocess to run with timeout
        centers, radii, width, height = run_with_timeout(
            program_path, timeout_seconds=timeout_duration
        )

        end_time = time.time()
        eval_time = end_time - start_time

        # Ensure centers and radii are numpy arrays
        if not isinstance(centers, np.ndarray):
            centers = np.array(centers)
        if not isinstance(radii, np.ndarray):
            radii = np.array(radii)

        # Check shape and size
        if centers.shape != (21, 2):
            shape_error = f"Invalid shapes: centers={centers.shape}, expected (21, 2)"
            print(shape_error)

            return {
                "status": "validation_failed",
                "summary": "Validation failed: The output 'centers' array has an incorrect shape.",
                "score": 0.0,
                "metrics": {
                    "sum_radii": 0.0,
                    "target_ratio": 0.0,
                    "validity": 0.0,
                    "eval_time": float(eval_time),
                },
                "artifacts": {
                    "stderr": shape_error,
                    "failure_stage": "shape_validation",
                    "expected_shapes": "centers: (21, 2)",
                    "actual_shapes": f"centers: {centers.shape}",
                    "execution_time": f"{eval_time:.2f}s",
                },
            }

        # Validate solution using check_construction_rectangle
        validation_result = check_construction_rectangle(
            centers, radii, 21, width, height
        )

        is_valid = validation_result.get("sum_of_radii", -np.inf) > -np.inf
        sum_radii = validation_result["sum_of_radii"] if is_valid else 0.0

        # Metrics calculation
        target_ratio = sum_radii / TARGET_VALUE if is_valid else 0.0
        validity = 1.0 if is_valid else 0.0
        combined_score = target_ratio * validity

        print(
            f"Evaluation: valid={is_valid}, sum_radii={sum_radii:.6f}, "
            f"target={TARGET_VALUE}, ratio={target_ratio:.6f}, time={eval_time:.2f}s"
        )

        # Prepare artifacts, status, and summary based on validation result
        artifacts = {"execution_time": f"{eval_time:.2f}s"}
        if not is_valid:
            status = "validation_failed"
            error_message = validation_result.get(
                "error_message", "Unknown validation error"
            )
            summary = f"Validation failed: {error_message}"
            artifacts["validation_report"] = f"Valid: False. Error: {error_message}"
            artifacts["validation_error"] = error_message
            artifacts["failure_stage"] = "geometric_validation"
        else:
            summary = f"Evaluation successful. The packing is valid with a total radii sum of {sum_radii:.6f}."
            artifacts["validation_report"] = (
                f"Valid: True. Packing verified successfully. Width={width:.4f}, Height={height:.4f}."
            )
            artifacts["packing_summary"] = (
                f"Sum of radii: {sum_radii:.6f}/{TARGET_VALUE} = {target_ratio:.4f}"
            )

            # Add successful packing stats for good solutions
            if target_ratio > 0.95:  # Near-optimal solutions
                artifacts["stdout"] = (
                    f"Excellent packing! Achieved {target_ratio:.1%} of target value"
                )
                min_radius = float(np.min(radii))
                max_radius = float(np.max(radii))
                avg_radius = float(np.mean(radii))
                artifacts["radius_stats"] = (
                    f"Min: {min_radius:.6f}, Max: {max_radius:.6f}, Avg: {avg_radius:.6f}"
                )

        return {
            "status": status,
            "summary": summary,
            "score": float(combined_score),
            "metrics": {
                "sum_radii": float(sum_radii),
                "target_ratio": float(target_ratio),
                "validity": float(validity),
                "eval_time": float(eval_time),
            },
            "artifacts": artifacts,
        }

    except TimeoutError as e:
        error_msg = f"Evaluation timed out: {str(e)}"
        print(error_msg)
        return {
            "status": "execution_failed",
            "summary": f"Execution failed: The program timed out after {timeout_duration} seconds.",
            "score": 0.0,
            "metrics": {
                "sum_radii": 0.0,
                "target_ratio": 0.0,
                "validity": 0.0,
                "eval_time": float(timeout_duration),
            },
            "artifacts": {
                "stderr": error_msg,
                "failure_stage": "execution_timeout",
                "timeout_duration": f"{timeout_duration}s",
                "suggestion": "Consider optimizing the packing algorithm for faster convergence",
            },
        }
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


"""
if __name__ == "__main__":
    file = "./initial_program.py"
    res = evaluate_stage1(file)
    print(f"{res}")
    res = evaluate_stage2(file)
    print(f"{res}")
"""
