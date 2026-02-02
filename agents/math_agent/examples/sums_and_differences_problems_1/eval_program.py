"""
Find a set of integers that maximizes the ratio log(|A+A|/|A|) / log(|A-A|/|A|).
Target value to beat is approximately 1.1219357374860444.
Enhanced with artifacts to demonstrate execution feedback.
"""

import importlib.util
import json
import math
import os
import sys
import time
import traceback

import numpy as np
from numba import njit


class TimeoutError(Exception):
    """Timeout error"""

    pass


def timeout_handler(signum, frame):
    """Handle timeout signal"""
    raise TimeoutError("Function execution timed out")


def get_score(best_list):
    """Returns the score for the given list using Numba."""
    if isinstance(best_list, np.ndarray):
        best_list = best_list.tolist()

    try:
        best_list = [int(x) for x in best_list]
    except (ValueError, TypeError):
        print(f"get_score List contains non-convertible values: {best_list}")
        return 0

    if len(best_list) < 2:
        print(f"get_score List is too short: {best_list}")
        return 0

    # if the list contains non-integers, return 0
    if not all(isinstance(x, int) for x in best_list):
        print(f"get_score List contains non-integers: {best_list}")
        return 0

    return get_score_numba(best_list) + (1.0 - 1.0 / len(set(best_list))) / 100.0


@njit
def get_score_numba(best_list):
    """Returns the score for the given list using Numba."""

    best_list_set = set(best_list)
    n_unique = len(best_list_set)

    a_minus_a = set()
    for i in best_list:
        for j in best_list:
            a_minus_a.add(i - j)

    a_plus_a = set()
    for i in best_list:
        for j in best_list:
            a_plus_a.add(i + j)

    lhs = len(a_minus_a) / n_unique
    rhs = len(a_plus_a) / n_unique

    try:
        return math.log(rhs) / math.log(lhs)
    except Exception:
        return 0


def run_with_timeout(program_path, timeout_seconds=20):
    """
    Run the program using dynamic import mechanism (Reference from Code 2).
    Directly calls search_for_best_set() and handles return values.
    Timeout logic is delegated to the external caller.
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

        if not hasattr(program_module, "search_for_best_set"):
            raise AttributeError(
                f"Function 'search_for_best_set' not found in {program_path}"
            )

        print("Calling search_for_best_set()...")

        returned_tuple = program_module.search_for_best_set()

        if isinstance(returned_tuple, tuple) and len(returned_tuple) >= 1:
            actual_best_list = returned_tuple[0]
        else:
            actual_best_list = returned_tuple

        print(f"search_for_best_set() returned successfully. List: {actual_best_list}")

        return actual_best_list

    except Exception as e:
        print(f"Error during execution: {str(e)}")
        traceback.print_exc()
        raise RuntimeError(f"Program execution failed: {str(e)}") from e

    finally:
        if program_dir in sys.path:
            sys.path.remove(program_dir)

        if module_name in sys.modules:
            del sys.modules[module_name]


def evaluate(program_path):
    """
    Evaluates the program by running it and checking the result.
    """
    # Target value from the paper
    TARGET_VALUE = 1.1319033750264975  # AlphaEvolve result
    start_time = time.time()

    try:
        best_list = run_with_timeout(program_path, timeout_seconds=3600)
        eval_time = time.time() - start_time

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

    if not isinstance(best_list, (list, np.ndarray, tuple)):
        summary = (
            f"Validation failed: Output type is incorrect. "
            f"Expected list or array, but got {type(best_list)}."
        )
        print(summary, file=sys.stderr)
        return {
            "status": "validation_failed",
            "summary": summary,
            "score": 0.0,
            "metrics": {"validity": 0.0},
            "artifacts": {
                "reason": "Invalid return type",
                "actual_type": str(type(best_list)),
            },
        }

    if len(best_list) == 0:
        summary = "Validation failed: Returned list is empty."
        print(summary, file=sys.stderr)
        return {
            "status": "validation_failed",
            "summary": summary,
            "score": 0.0,
            "metrics": {"validity": 0.0},
            "artifacts": {"reason": "Empty list"},
        }

    print("DEBUG: Best list to get score:", best_list)
    calculated_score = get_score(best_list)

    if calculated_score == 0:
        summary = "Validation failed: Calculated score is 0. List may contain non-integers or be too short."
        print(summary, file=sys.stderr)
        return {
            "status": "validation_failed",
            "summary": summary,
            "score": 0.0,
            "metrics": {"validity": 0.0},
            "artifacts": {"reason": "Invalid content in list"},
        }

    is_success = calculated_score > TARGET_VALUE
    target_ratio = calculated_score / TARGET_VALUE

    if is_success:
        summary = f"Success: New best set found! Score: {calculated_score:.6f} > Target: {TARGET_VALUE:.6f}"
    else:
        summary = f"Completed: Valid set found, \
but score did not beat target. Score: {calculated_score:.12f} <= Target: {TARGET_VALUE:.12f}"

    print(summary)

    # Convert list to standard python list for JSON serialization
    if isinstance(best_list, np.ndarray):
        serializable_list = best_list.tolist()
    else:
        serializable_list = list(best_list)

    return {
        "status": "success",
        "summary": summary,
        "score": float(target_ratio),
        "metrics": {
            "get_score_result": float(calculated_score),
            "target_value": float(TARGET_VALUE),
            "validity": 1.0,
            "eval_time": float(eval_time),
        },
        "artifacts": {
            "execution_time": f"{eval_time:.2f}s",
            "best_list": json.dumps(serializable_list, ensure_ascii=False),
            "list_length": len(serializable_list),
        },
    }
