"""
Evaluator for circle packing example (n=26) with improved timeout handling
Enhanced with artifacts to demonstrate execution feedback
"""

import os
import pickle
import subprocess
import sys
import tempfile
import time
import traceback

import numpy as np


class TimeoutError(Exception):
    pass


def timeout_handler(signum, frame):
    """Handle timeout signal"""
    raise TimeoutError("Function execution timed out")


def validate_generating(step_function_heights_1):
    validation_details = {
        "has_negative_val": [],
    }

    for val in step_function_heights_1:
        if val < 0:
            violation = f"Value {val} is negative"
            validation_details["has_negative_val"].append(violation)
            print(violation)

    is_valid = len(validation_details["has_negative_val"]) == 0
    validation_details["is_valid"] = is_valid

    return is_valid, validation_details


def cal_lower_bound(convolution_2: list[float]):
    import numpy as np

    # Calculate the 2-norm squared: ||f*f||_2^2
    num_points = len(convolution_2)
    x_points = np.linspace(-0.5, 0.5, num_points + 2)
    x_intervals = np.diff(x_points)  # Width of each interval
    y_points = np.concatenate(([0], convolution_2, [0]))
    l2_norm_squared = 0.0
    for i in range(len(convolution_2) + 1):  # Iterate through intervals
        y1 = y_points[i]
        y2 = y_points[i + 1]
        h = x_intervals[i]
        # Integral of (mx + c)^2 = h/3 * (y1^2 + y1*y2 + y2^2) where m = (y2-y1)/h, c = y1 - m*x1, interval is [x1, x2], y1 = mx1+c, y2=mx2+c
        interval_l2_squared = (h / 3) * (y1**2 + y1 * y2 + y2**2)
        l2_norm_squared += interval_l2_squared

    # Calculate the 1-norm: ||f*f||_1
    norm_1 = np.sum(np.abs(convolution_2)) / (len(convolution_2) + 1)

    # Calculate the infinity-norm: ||f*f||_inf
    norm_inf = np.max(np.abs(convolution_2))
    C_lower_bound = l2_norm_squared / (norm_1 * norm_inf)

    print(f"This step function shows that C2 >= {C_lower_bound}")
    return C_lower_bound


def run_with_timeout(program_path, timeout_seconds=20):
    """
    Run the program in a separate process with timeout
    using a simple subprocess approach

    Args:
        program_path: Path to the program file
        timeout_seconds: Maximum execution time in seconds

    Returns:
        centers, radii, sum_radii tuple from the program
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

    # Run the generate function
    print("Calling generate()...")
    step_function_heights_1, c_lower_bound = program.optimize_lower_bound()
    print(f"generate() returned successfully")

    # Save results to a file
    results = {{
        'step_function_heights_1': step_function_heights_1,
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

                return results["step_function_heights_1"]
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
    Evaluate the program by running it once and checking the C_upper_bound

    Args:
        program_path: Path to the program file

    Returns:
        EvaluationResult with metrics and artifacts
    """
    # Target value from the paper
    TARGET_VALUE = 0.8962  # AlphaEvolve result

    try:
        # For constructor-based approaches, a single evaluation is sufficient
        # since the result is deterministic
        start_time = time.time()

        # Use subprocess to run with timeout
        step_function_heights_1 = run_with_timeout(
            program_path, timeout_seconds=3600  # Single timeout
        )

        convolution_2 = np.convolve(step_function_heights_1, step_function_heights_1)
        C_lower_bound = cal_lower_bound(convolution_2)

        end_time = time.time()
        eval_time = end_time - start_time

        # Validate solution
        valid, validation_details = validate_generating(step_function_heights_1)

        # Check shape and size
        shape_valid = step_function_heights_1.shape == (50,)
        if not shape_valid:
            shape_error = f"Invalid shapes: step_function_heights_1={step_function_heights_1.shape}, expected (50,)"
            print(shape_error)

            return {
                "score": 0.0,
                "metrics": {
                    "C_lower_bound": 0.0,
                    "target_ratio": 0.0,
                    "validity": 0.0,
                    "eval_time": float(eval_time),
                },
                "artifacts": {
                    "stderr": shape_error,
                    "failure_stage": "shape_validation",
                    "expected_shapes": "step_function_shapes: (50,)",
                    "actual_shapes": f"step_function_shapes: {step_function_heights_1.shape}",
                    "execution_time": f"{eval_time:.2f}s",
                },
            }

        # Target ratio (how close we are to the target)
        target_ratio = C_lower_bound / TARGET_VALUE if valid else 0.0

        # Validity score
        validity = 1.0 if valid else 0.0

        # Combined score - higher is better
        combined_score = target_ratio * validity

        print(
            f"Evaluation: valid={valid}, C_lower_bound={C_lower_bound:.6f}, target={TARGET_VALUE}, ratio={target_ratio:.6f}, time={eval_time:.2f}s"
        )

        # Prepare artifacts with packing details
        artifacts = {
            "execution_time": f"{eval_time:.2f}s",
            "packing_summary": f"C_lower_bound: {C_lower_bound:.6f}/{TARGET_VALUE} = {target_ratio:.4f}",
            "validation_report": f"Valid: {valid}, Violations: {len(validation_details.get('has_negative_val', []))}",
        }

        # Add validation details if there are issues
        if not valid:
            if validation_details.get("has_negative_val"):
                artifacts["has_negative_val"] = "\n".join(
                    validation_details["has_negative_val"]
                )

        # Add successful packing stats for good solutions
        if valid and target_ratio > 0.98:  # Near-optimal solutions
            artifacts["stdout"] = (
                f"Excellent generating! Achieved {target_ratio:.1%} of target value"
            )
            artifacts["C_lower_bound_stats"] = f"C_lower_bound: {C_lower_bound:.6f}"

        return {
            "score": float(combined_score),
            "metrics": {
                "C_lower_bound": float(C_lower_bound),
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
            "score": 0.0,
            "metrics": {
                "C_lower_bound": 0.0,
                "target_ratio": 0.0,
                "validity": 0.0,
                "eval_time": 1800.0,  # Timeout duration
            },
            "artifacts": {
                "stderr": error_msg,
                "failure_stage": "execution_timeout",
                "timeout_duration": "1800s",
                "suggestion": "Consider optimizing the generate algorithm for faster convergence",
            },
        }
    except Exception as e:
        error_msg = f"Evaluation failed completely: {str(e)}"
        print(error_msg)
        traceback.print_exc()
        return {
            "score": 0.0,
            "metrics": {
                "C_lower_bound": 0.0,
                "target_ratio": 0.0,
                "validity": 0.0,
                "eval_time": 0.0,
            },
            "artifacts": {
                "stderr": error_msg,
                "traceback": traceback.format_exc(),
                "failure_stage": "program_execution",
                "suggestion": "Check for syntax errors, import issues, or runtime exceptions",
            },
        }


# Stage-based evaluation for cascade evaluation
def evaluate_stage1(program_path):
    """
    First stage evaluation - quick validation check
    Enhanced with artifacts for debugging
    """
    try:
        # Use the simplified subprocess approach
        try:
            start_time = time.time()
            step_function_heights_1 = run_with_timeout(
                program_path, timeout_seconds=1800
            )
            eval_time = time.time() - start_time

            convolution_2 = np.convolve(
                step_function_heights_1, step_function_heights_1
            )
            C_lower_bound = cal_lower_bound(convolution_2)

            # Validate solution (shapes and constraints)
            shape_valid = step_function_heights_1.shape == (50,)
            if not shape_valid:
                shape_error = f"Invalid shapes: step_function_heights_1={step_function_heights_1.shape}"
                print(shape_error)
                return {
                    "score": 0.0,
                    "metrics": {"validity": 0.0},
                    "artifacts": {
                        "stderr": shape_error,
                        "failure_stage": "stage1_shape_validation",
                        "expected_shapes": "step_function_shape: (50,)",
                        "actual_shapes": f"step_function_shape: {step_function_heights_1.shape}",
                        "execution_time": f"{eval_time:.2f}s",
                    },
                }

            valid, validation_details = validate_generating(step_function_heights_1)

            # Target from paper
            target = 0.8962

            # Simple combined score for stage 1
            combined_score = (C_lower_bound / target) if valid else 0.0

            # Prepare artifacts for stage 1
            artifacts = {
                "execution_time": f"{eval_time:.2f}s",
                "stage": "quick_validation",
                "packing_summary": f"C_lower_bound: {C_lower_bound:.6f}, Ratio: {C_lower_bound/target:.4f}",
            }

            # Add validation issues if any
            if not valid:
                artifacts["stderr"] = (
                    f"Validation failed: {len(validation_details.get('has_negative_val', []))}"
                )
                artifacts["failure_stage"] = "stage1_geometric_validation"
                if validation_details.get("has_negative_val"):
                    artifacts["has_negative_val"] = validation_details[
                        "has_negative_val"
                    ][
                        0
                    ]  # Just first issue

            # Return evaluation metrics
            return {
                "score": float(combined_score),
                "metrics": {
                    "validity": 1.0 if valid else 0.0,
                    "C_lower_bound": float(C_lower_bound),
                    "target_ratio": float(C_lower_bound / target if valid else 0.0),
                },
                "artifacts": artifacts,
            }

        except TimeoutError as e:
            error_msg = f"Stage 1 evaluation timed out: {e}"
            print(error_msg)
            return {
                "score": 0.0,
                "metrics": {"validity": 0.0},
                "artifacts": {
                    "stderr": error_msg,
                    "failure_stage": "stage1_timeout",
                    "timeout_duration": "1800s",
                    "suggestion": "Algorithm may be too slow for stage 1 - consider simpler heuristics",
                },
            }
        except Exception as e:
            error_msg = f"Stage 1 evaluation failed: {e}"
            print(error_msg)
            print(traceback.format_exc())
            return {
                "score": 0.0,
                "metrics": {"validity": 0.0},
                "artifacts": {
                    "stderr": error_msg,
                    "traceback": traceback.format_exc(),
                    "failure_stage": "stage1_execution",
                    "suggestion": "Check basic syntax and imports before attempting full evaluation",
                },
            }

    except Exception as e:
        error_msg = f"Stage 1 evaluation failed completely: {e}"
        print(error_msg)
        print(traceback.format_exc())
        return {
            "score": 0.0,
            "metrics": {"validity": 0.0},
            "artifacts": {
                "stderr": error_msg,
                "traceback": traceback.format_exc(),
                "failure_stage": "stage1_critical_failure",
                "suggestion": "Major issues detected - check program structure and dependencies",
            },
        }


def evaluate_stage2(program_path):
    """
    Second stage evaluation - full evaluation
    """
    # Full evaluation as in the main evaluate function
    return evaluate(program_path)
