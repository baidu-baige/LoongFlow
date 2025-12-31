"""
Evaluator for the first_autocorrelation_inequality, aligned with standard specifications.
"""

import importlib.util
import os
import sys
import time
import traceback

import numpy as np

TARGET_VALUE = (
    1.5053  # sota of a step function with 600 equally-spaced intervals on [-1/4, 1/4]
)
TIMEOUT_SECONDS = 1000


class TimeoutError(Exception):
    """Custom timeout exception."""

    pass


def evaluate_sequence(sequence: list[float]):
    """
    Evaluates a sequence of coefficients with enhanced security checks.
    Returns np.inf if the input is invalid.
    """
    # --- Security Checks ---

    # Verify that the input is a list
    if not isinstance(sequence, list):
        return np.inf, "Sequence must be a list"

    # Reject empty lists
    if not sequence:
        return np.inf, "Sequence must not be empty"

    # Check each element in the list for validity
    for x in sequence:
        # Reject boolean types (as they are a subclass of int) and
        # any other non-integer/non-float types (like strings or complex numbers).
        if isinstance(x, bool) or not isinstance(x, (int, float)):
            return np.inf, "value in the sequence must be integers or floats"

        # Reject Not-a-Number (NaN) and infinity values.
        if np.isnan(x) or np.isinf(x):
            return np.inf, "value in the sequence must not be NaN or infinite"

    # Convert all elements to float for consistency
    sequence = [float(x) for x in sequence]

    # Protect against negative numbers
    sequence = [max(0, x) for x in sequence]

    # Protect against numbers that are too large
    sequence = [min(1000.0, x) for x in sequence]

    n = len(sequence)
    b_sequence = np.convolve(sequence, sequence)
    max_b = max(b_sequence)
    sum_a = np.sum(sequence)

    # Protect against the case where the sum is too close to zero
    if sum_a < 0.01:
        return (
            np.inf,
            "Sum of sequence is too close to zero, it must be greater than 0.01",
        )

    return float(2 * n * max_b / (sum_a**2)), ""


def run_with_timeout(program_path, timeout_seconds=TIMEOUT_SECONDS):
    """
    Runs the program in a separate process with a timeout.
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

        if not hasattr(program_module, "run_search_for_best_sequence"):
            raise AttributeError(
                f"Function 'run_search_for_best_sequence' not found in {program_path}"
            )

        print(f"Calling run_search_for_best_sequence()...")

        best_sequence = program_module.run_search_for_best_sequence()

        print(f"run_search_for_best_sequence() returned successfully")
        return best_sequence

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
    Evaluates the program and returns a structured result dictionary.
    """
    start_time = time.time()
    status = "success"

    try:
        best_sequence = run_with_timeout(program_path)
        eval_time = time.time() - start_time

        if not isinstance(best_sequence, list):
            best_sequence = list(best_sequence)

        # --- Geometric Validation ---
        c1, error_message = evaluate_sequence(best_sequence)
        validity = 1.0 if error_message == "" else 0.0

        # If valid, use the returned c1. Otherwise, it's 0.
        target_ratio = TARGET_VALUE / c1 if error_message == "" else 0.0

        # The final score is the target ratio, penalized if invalid.
        score = target_ratio * validity

        artifacts = {"execution_time": f"{eval_time:.2f}s"}
        if error_message != "":
            status = "validation_failed"
            summary = f"Validation failed: {error_message}"
            artifacts["validation_report"] = f"Validation failed: {error_message}"
            artifacts["failure_stage"] = "geometric_validation"
        else:
            status = "success"
            summary = f"Evaluation successful."
            artifacts["validation_report"] = "Best sequence is valid."
            artifacts["summary"] = f"Achieved {target_ratio:.2%} of benchmark."

        return {
            "score": float(score),
            "status": status,
            "summary": summary,
            "metrics": {
                "upper_bound": float(c1),
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
                "min_area": 0.0,
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
                "min_area": 0.0,
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
    print(f"{res}")
