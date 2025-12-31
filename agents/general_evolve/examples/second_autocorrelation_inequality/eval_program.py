"""
Optimize Second autocorrelation inequality Problem
"""
import importlib.util
import os
import sys
import time
import traceback

import numpy as np

TIMEOUT = 3600

class TimeoutError(Exception):
    """Raised when a timeout occurs."""
    pass


def timeout_handler(signum, frame):
    """Handle timeout signal"""
    raise TimeoutError("Function execution timed out")

def verify_heights_sequence(heights_sequence_2: np.ndarray, c_lower_bound: float):
    """verify_heights_sequence(heights_sequence_2, c_lower_bound)"""
    if len(heights_sequence_2) != 50:
        return False, f"len(heights_sequence_2) not 50"

    for i in range(len(heights_sequence_2)):
        if heights_sequence_2[i] < 0:
            return False, f"heights_sequence_2 all elements must be non-negative"

    convolution_2 = np.convolve(heights_sequence_2, heights_sequence_2)
    c_c_lower_bound = cal_lower_bound(convolution_2)
    if c_lower_bound != c_c_lower_bound:
        return False, f"c_lower_bound: {c_lower_bound} miscalculation, the correct value is {c_c_lower_bound}"

    return True, ""

def cal_lower_bound(convolution_2: list[float]):
    """cal_lower_bound(convolution_2)"""
    # Calculate the 2-norm squared: ||f*f||_2^2
    num_points = len(convolution_2)
    x_points = np.linspace(-0.5, 0.5, num_points + 2)
    x_intervals = np.diff(x_points) # Width of each interval
    y_points = np.concatenate(([0], convolution_2, [0]))
    l2_norm_squared = 0.0
    for i in range(len(convolution_2) + 1):  # Iterate through intervals
        y1 = y_points[i]
        y2 = y_points[i + 1]
        h = x_intervals[i]
        interval_l2_squared = (h / 3) * (y1**2 + y1 * y2 + y2**2)
        l2_norm_squared += interval_l2_squared

    # Calculate the 1-norm: ||f*f||_1
    norm_1 = np.sum(np.abs(convolution_2)) / (len(convolution_2) + 1)

    # Calculate the infinity-norm: ||f*f||_inf
    norm_inf = np.max(np.abs(convolution_2))
    c_lower_bound = l2_norm_squared / (norm_1 * norm_inf)

    print(f"This step function shows that C2 >= {c_lower_bound}")
    return c_lower_bound

def run_with_timeout(program_path, timeout_seconds=20):
    """
    Run the program using its existing unique filename.
    Supports multiprocessing naturally.

    """
    # 1. Parse the path to get the directory and the filename without extension
    program_dir, file_name = os.path.split(program_path)
    module_name, _ = os.path.splitext(file_name)

    # --- Security Check ---
    # Check if the module name is a valid Python identifier
    if not module_name.isidentifier():
        raise ValueError(
            f"Invalid module name: '{module_name}'. "
            "Filename must contain only letters, numbers, and underscores, "
            "and cannot start with a number."
        )

    # 2. Add the current file’s directory to sys.path
    # This ensures that both the main process and subprocesses can find the file
    if program_dir not in sys.path:
        sys.path.insert(0, program_dir)

    try:
        # 3. Dynamically import a module
        # If sys.modules contains stale data (highly unlikely due to unique random filenames), reload the module first
        if module_name in sys.modules:
            program_module = importlib.reload(sys.modules[module_name])
        else:
            program_module = importlib.import_module(module_name)

        if not hasattr(program_module, 'optimize_lower_bound'):
            raise AttributeError(f"Function 'optimize_lower_bound' not found in {program_path}")

        print(f"Calling optimize_lower_bound()...")

        # 4. Execute the code
        # Since module_name corresponds to a unique, physical file on disk, and its path is included in sys.path
        # so multiprocessing's subprocess can perfectly find and load this module
        heights_sequence_2, c_lower_bound = program_module.optimize_lower_bound()

        print(f"optimize_lower_bound() returned successfully: c_lower_bound = {c_lower_bound}")
        return heights_sequence_2, c_lower_bound

    except Exception as e:
        print(f"Error during execution: {str(e)}")
        raise RuntimeError(f"Program execution failed: {str(e)}") from e

    finally:
        # 5. Cleanup work (crucial)

        # A. Remove the program's directory from sys.path
        if program_dir in sys.path:
            sys.path.remove(program_dir)

        # B. Remove the program's module from sys.modules avoid memory leaking
        # This is crucial to avoid memory leaks in case the program fails to exit gracefully
        if module_name in sys.modules:
            del sys.modules[module_name]


def evaluate(program_path):
    """
    Evaluate the program by running it once and checking

    Args:
        program_path: Path to the program file

    Returns:
        EvaluationResult with metrics and artifacts
    """
    # Target value from the paper
    TARGET_VALUE = 0.8963
    start_time = time.time()

    try:
        # For constructor-based approaches, a single evaluation is sufficient
        # since the result is deterministic
        # Use subprocess to run with timeout
        heights_sequence_2, c_lower_bound = run_with_timeout(
            program_path, timeout_seconds=3600  # Single timeout
        )

        # Ensure centers and radii are numpy arrays
        if not isinstance(heights_sequence_2, np.ndarray):
            heights_sequence_2 = np.array(heights_sequence_2)

        # Validate solution
        valid, error = verify_heights_sequence(heights_sequence_2, c_lower_bound)
        if not valid:
            return {
                "status": "validation_failed",
                "score": 0.0,
                "summary": error,
                "metrics": {"validity": 0.0, "c_lower_bound": 0.0},
                "artifacts": {"reason": "Geometric constraints not met."},
            }

        # Target ratio (how close we are to the target)
        target_ratio = c_lower_bound / TARGET_VALUE if valid else 0.0

        # Validity score
        validity = 1.0 if valid else 0.0

        # Combined score - higher is better
        combined_score = target_ratio * validity

        return {
            "status": "success",
            "summary": f"Success: Valid lower_bound found. lower_bound = {c_lower_bound}, Score: {combined_score:.4f}",
            "score": float(combined_score),
            "metrics": {
                "c_lower_bound": float(c_lower_bound),
                "target_ratio": float(target_ratio),
                "validity": float(validity),
                "eval_time": float(time.time() - start_time),
            }
        }

    except TimeoutError as e:
        eval_time = time.time() - start_time
        error_msg = f"Evaluation timed out: {str(e)}"
        print(error_msg)
        return {
            "score": 0.0,
            "status": "execution_failed",
            "summary": f"Execution failed: The program timed out after {TIMEOUT} seconds.",
            "metrics": {"c_lower_bound": 0.0, "target_ratio": 0.0, "validity": 0.0, "eval_time": eval_time},
            "artifacts": {
                "stderr": str(e),
                "failure_stage": "execution_timeout",
                "execution_time": f"{eval_time:.2f}s",
            },
        }
    except Exception as e:
        error_msg = f"Evaluation failed completely: {str(e)}"
        return {
            "score": 0.0,
            "status": "execution_failed",
            "summary": f"Execution failed: {str(e)}",
            "metrics": {
                "c_lower_bound": 0.0,
                "target_ratio": 0.0,
                "validity": 0.0,
                "eval_time": 0.0,
                "combined_score": 0.0,
            },
            "artifacts": {
                "stderr": error_msg,
                "traceback": traceback.format_exc(),
                "failure_stage": "program_execution",
                "suggestion": "Check for syntax errors, import issues, or runtime exceptions",
            },
        }


def evaluate_stage1(program_path):
    """
    First stage evaluation - quick validation check
    Enhanced with artifacts for debugging
    """
    # Full evaluation as in the main evaluate function
    return evaluate(program_path)

if __name__ == "__main__":
    file = "./initial.py"
    evaluate_stage1(file)