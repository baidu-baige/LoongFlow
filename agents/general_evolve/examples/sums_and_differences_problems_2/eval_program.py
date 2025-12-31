"""
Find a set of integers that maximizes the ratio log(|A+A|/|A|) / log(|A-A|/|A|).
Target value to beat is approximately 1.1479888965092757.
Enhanced with artifacts to demonstrate execution feedback.
"""

import importlib.util
import json
import os
import pickle
import subprocess
import sys
import tempfile
import time
import traceback
import signal
from typing import Union

import numpy as np


class TimeoutError(Exception):
    """Timeout error"""
    pass


def timeout_handler(signum, frame):
    """Handle timeout signal"""
    raise TimeoutError("Function execution timed out")


def compute_lower_bound(u: list[int]) -> Union[float, str]:
    """
    Returns the lower bound obtained from the input set u, which must satisfy min(u) == 0.
    If a constraint is violated, returns an error message string instead of raising an exception.
    """
    # 1. Compatibility handling: convert numpy array to list
    if isinstance(u, np.ndarray):
        u = u.tolist()

    # 2. Compatibility handling: ensure elements are integers
    try:
        u = [int(x) for x in u]
    except (ValueError, TypeError):
        return f"Error: Input list contains non-convertible values: {u}"
        
    # Check if the input list is empty to avoid errors in min/max
    if not u:
        return "Error: Input set U cannot be empty."

    if 0 not in u:
        u = [0] + u
    # if min(u) != 0:
    #     return f"Error: Set U must be nonnegative and must contain 0; got minimum value {min(u)}"

    max_u = max(u)
    u_minus_u = np.zeros(2 * max_u + 1, dtype=bool)  # Store the set u - u as an array of booleans.
    u_plus_u = np.zeros(2 * max_u + 1, dtype=bool)  # Store the set u + u as an array of booleans.
    u_np = np.array(u)

    for i in u:
        u_minus_u[i - u_np + max_u] = True
        u_plus_u[i + u_np] = True

    u_minus_u_size = np.sum(u_minus_u)
    u_plus_u_size = np.sum(u_plus_u)

    if u_minus_u_size > 2 * max_u + 1:
        return (
            "Error: The constraint |U - U| <= 2 max (U) + 1 is not satisfied. Got: "
            f"lhs={u_minus_u_size} but rhs={2 * max_u + 1}."
        )

    try:
        val = np.log(u_minus_u_size / u_plus_u_size) / np.log(2 * max_u + 1) + 1.0
        return float(val)
    except Exception as e:
        return f"Error during calculation: {str(e)}"


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
        # Try a simple replacement fix in case filename contains '-' etc.
        safe_module_name = module_name.replace('-', '_').replace(' ', '_')
        if not safe_module_name.isidentifier():
             raise ValueError(
                f"Invalid module name: '{module_name}'. "
                "Filename must be a valid Python identifier."
            )
        module_name = safe_module_name

    # 2. Environment Preparation
    if program_dir not in sys.path:
        sys.path.insert(0, program_dir)

    # Set timeout signal
    # Note: signal.SIGALRM is only valid on Unix/Linux/Mac. Windows requires a different approach.
    if hasattr(signal, 'SIGALRM'):
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout_seconds)

    try:
        # 3. Dynamic Import
        if module_name in sys.modules:
            # If the module exists, force reload to ensure the latest code is run
            program_module = importlib.reload(sys.modules[module_name])
        else:
            # Use spec_from_file_location to load from arbitrary path
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            if spec is None or spec.loader is None:
                 raise ImportError(f"Could not load spec for file: {file_path}")
            program_module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = program_module
            spec.loader.exec_module(program_module)

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
        if hasattr(signal, 'SIGALRM'):
            signal.alarm(0) # Cancel the alarm

        if program_dir in sys.path:
            sys.path.remove(program_dir)

        # Only consider cleaning up if we actually loaded or reloaded the module.
        if module_name in sys.modules: 
            del sys.modules[module_name]


def evaluate(program_path):
    """
    Evaluates the program by running it and checking the result using compute_lower_bound.
    """
    # Target value updated as requested
    # 54265
    TARGET_VALUE = 1.17 # 2003 elements: 1.1479888965092757
    start_time = time.time()

    try:
        # 1. --- Execution Phase ---
        # Replaced run_with_timeout with run_external_function
        returned_data = run_external_function(
            program_path, 
            "search_for_best_set", 
            timeout_seconds=3600
        )
        
        # Handle the return value (Tuple compatibility logic)
        if isinstance(returned_data, tuple) and len(returned_data) >= 1:
            best_list = returned_data[0]  # Extract array/list
            message = returned_data[1] if len(returned_data) > 1 else ""
            if message:
                print(f"Message from function: {message}")
        else:
            # Compatibility for cases returning only a list
            best_list = returned_data
            
        eval_time = time.time() - start_time

    except Exception as e:
        # --- Scenario 1: Execution Failed ---
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
    # Validate if return type is list or array
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
                "actual_type": str(type(best_list))
            }
        }
    
    # Ensure it is not empty
    if len(best_list) == 0:
        summary = "Validation failed: Returned list is empty."
        print(summary, file=sys.stderr)
        return {
            "status": "validation_failed",
            "summary": summary,
            "score": 0.0,
            "metrics": {"validity": 0.0},
            "artifacts": {"reason": "Empty list"}
        }

    # 3. --- Scoring Phase ---
    # Use compute_lower_bound to calculate score
    print("DEBUG: Best list to compute lower bound:", best_list)
    
    # Call compute_lower_bound
    lower_bound_result = compute_lower_bound(best_list)
    
    # Check compute_lower_bound result
    if isinstance(lower_bound_result, str):
        # If string is returned, it means an error or constraint violation occurred
        summary = f"Validation failed: compute_lower_bound returned error: {lower_bound_result}"
        print(summary, file=sys.stderr)
        return {
            "status": "validation_failed",
            "summary": summary,
            "score": 0.0,
            "metrics": {"validity": 0.0},
            "artifacts": {"reason": "Constraint violation or calc error", "error_msg": lower_bound_result}
        }

    calculated_score = float(lower_bound_result)

    # 4. --- Success Phase ---
    is_success = calculated_score > TARGET_VALUE
    target_ratio = calculated_score / TARGET_VALUE
    
    if is_success:
        summary = f"Success: New best set found! Lower Bound: {calculated_score:.12f} > Target: {TARGET_VALUE:.12f}"
    else:
        summary = f"Completed: Valid set found, \
but lower bound did not beat target. Result(get_score() result): {calculated_score:.12f} <= Target: {TARGET_VALUE:.12f}"

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
            "compute_lower_bound_result(get_score() result)": float(calculated_score),
            "target_value": float(TARGET_VALUE),
            "validity": 1.0,
            "eval_time": float(eval_time),
        },
        "artifacts": {
            "execution_time": f"{eval_time:.2f}s",
            "list_length": len(serializable_list)
        }
    }
