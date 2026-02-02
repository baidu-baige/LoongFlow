"""
Find n points in d-dimensional Euclidean space that minimize the ratio R = D_max / D_min,
where D_max is the maximum Euclidean distance between any two points,
and D_min is the minimum distance between any two distinct points in the set.
Enhanced with artifacts to demonstrate execution feedback
"""

import json
import os
import pickle
import subprocess
import sys
import tempfile
import time
import traceback
import signal
import importlib

import numpy as np
import scipy as sp
import scipy.spatial 

N = 16
D = 2

class TimeoutError(Exception):
    """Timeout error"""
    pass


def timeout_handler(signum, frame):
    """Handle timeout signal"""
    raise TimeoutError("Function execution timed out")


def verification(points: np.ndarray):
    """Verify the correctness of the generated points"""
    pairwise_distances = sp.spatial.distance.pdist(points)
    # Handle the case where there are no valid distances (e.g., all points are identical)
    if pairwise_distances.size == 0:
        return False, None

    min_distance = np.min(pairwise_distances)
    max_distance = np.max(pairwise_distances)
    if abs(min_distance) < 1e-10 or abs(max_distance) < 1e-10:
        return False, None
    
    ratio_squared = (max_distance / min_distance)**2
    if ratio_squared is None or ratio_squared < 1e-10:
        return False, None

    return True, ratio_squared


def run_external_function(file_path, func_name, timeout_seconds=20, *args, **kwargs):
    """
    Dynamically loads a Python file from a specified path and executes a specific function within it
    under a timeout constraint.
    
    Args:
        file_path (str): The full path to the target Python file.
        func_name (str): The name of the function to execute.
        timeout_seconds (int): Timeout duration in seconds.
        *args: Positional arguments to pass to the target function.
        **kwargs: Keyword arguments to pass to the target function.

    Returns:
        Any: The return value of the target function.
    
    Raises:
        TimeoutError: If execution times out.
        ValueError: If the filename is invalid.
        AttributeError: If the function does not exist.
        RuntimeError: If an error occurs during target code execution.
        FileNotFoundError: If the file path does not exist.
    """
    
    # 1. Path and Module Name Processing
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
        
    program_dir, file_name = os.path.split(os.path.abspath(file_path))
    module_name, _ = os.path.splitext(file_name)

    # Security check: Ensure the module name is a valid Python identifier
    if not module_name.isidentifier():
        raise ValueError(f"Invalid module name: '{module_name}'. Filename must be a valid Python identifier.")

    # 2. Environment Preparation
    if program_dir not in sys.path:
        sys.path.insert(0, program_dir)

    # Set timeout signal
    # Note: signal.SIGALRM is only valid on Unix/Linux/Mac. Windows requires a different approach.
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout_seconds)

    try:
        # 3. Dynamic Import
        if module_name in sys.modules:
            # If the module exists, force reload to ensure the latest code is run
            program_module = importlib.reload(sys.modules[module_name])
        else:
            program_module = importlib.import_module(module_name)

        # 4. Get Function
        if not hasattr(program_module, func_name):
            raise AttributeError(f"Function '{func_name}' not found in {file_path}")
        
        target_func = getattr(program_module, func_name)

        # 5. Execute Function (passing arguments)
        print(f"Executing {module_name}.{func_name} with timeout {timeout_seconds}s...")
        result = target_func(*args, **kwargs)
        
        return result

    except TimeoutError:
        raise TimeoutError(f"Execution timed out after {timeout_seconds} seconds")
    except Exception as e:
        # Catch all other exceptions, wrap and re-raise, preserving the original stack trace
        raise RuntimeError(f"Error executing {func_name}: {str(e)}") from e

    finally:
        # 6. Cleanup (Crucial to prevent environment pollution)
        signal.alarm(0) # Cancel the alarm

        if program_dir in sys.path:
            sys.path.remove(program_dir)

        # Only consider cleaning up if we actually loaded or reloaded the module.
        # Strategy: To ensure complete isolation for the next run, deletion is usually recommended.
        if module_name in sys.modules: 
            del sys.modules[module_name]


def evaluate(program_path):
    """
    Evaluates the program by running it and checking the result.
    """
    # Target value from the paper
    TARGET_VALUE = 12.889266112  # AlphaEvolve result for n=16
    start_time = time.time()
    status = "success"

    try:
        # 1. --- Execution Phase ---
        # Use run_external_function instead of run_with_timeout
        # Assume the target function is 'optimize_construct', which accepts n and d as parameters
        # and returns a tuple (points, ratio_squared)
        
        # Call the external function directly
        result = run_external_function(
            file_path=program_path,
            func_name="optimize_construct",
            timeout_seconds=3600,
            n=N,
            d=D
        )
        
        # Unpack the result (expecting tuple: points, ratio_squared)
        if isinstance(result, tuple) and len(result) >= 1:
            points = result[0]
        else:
            # Fallback if the user function just returns existing points
            points = result

        eval_time = time.time() - start_time

        # Ensure type is correct for validation
        if not isinstance(points, np.ndarray):
            points = np.array(points)

    except Exception as e:
        # --- Scenario 1: Execution Failed ---
        # Capture any possible errors in run_external_function
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
    # Check basic shape
    if points.shape != (16, 2):
        summary = (
            f"Validation failed: Output shape is incorrect. "
            f"Expected points shape: (16, 2), but got {points.shape}."
        )
        print(summary, file=sys.stderr)
        return {
            "status": "validation_failed",
            "summary": summary,
            "score": 0.0,
            "metrics": {"validity": 0.0},
            "artifacts": {
                "reason": "Invalid shape",
                "expected_shape": "points: (16, 2)",
                "actual_shape": f"points: {points.shape}"
            }
        }

    # Use verification function for strict geometric validation
    is_valid, ratio_squared = verification(points)
    if not is_valid:
        summary = "Validation failed: The generated points have invalid geometric " \
"properties (e.g., D_min is zero, non-finite values, or other issues)."
        print(summary, file=sys.stderr)
        return {
            "status": "validation_failed",
            "summary": summary,
            "score": 0.0,
            "metrics": {"validity": 0.0, "ratio_squared": float(ratio_squared) if ratio_squared is not None else 0.0},
            "artifacts": {"reason": "Geometric constraints not met."}
        }
    
    # 3. --- Success Phase ---
    # If the code runs to here, execution and validation are successful
    target_ratio = TARGET_VALUE / ratio_squared
    # The score can simply be target_ratio
    score = target_ratio

    summary = f"Success: Valid construction found. Ratio squared: {ratio_squared:.6f}, Score: {score:.4f}"
    print(summary)

    return {
        "status": status,
        "summary": summary,
        "score": float(score),
        "metrics": {
            "ratio_squared": float(ratio_squared),
            "target_ratio": float(target_ratio),
            "validity": 1.0,
            "eval_time": float(eval_time),
        },
        "artifacts": {
            "execution_time": f"{eval_time:.2f}s",
            "points": json.dumps(points.tolist(), ensure_ascii=False),
        }
    }
