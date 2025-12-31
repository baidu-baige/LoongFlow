"""
Evaluates the Erdős minimum overlap problem solution.
Finds a sequence of coefficients (0 or 1, relaxed to [0,1]) such that the
maximum overlap of the sequence with its complement is minimized.
"""

import importlib
import os
import signal
import sys
import time
import traceback

import numpy as np

TARGET_VALUE = 0.380927


class TimeoutError(Exception):
    """Timeout error"""

    pass


def timeout_handler(signum, frame):
    """Handle timeout signal"""
    raise TimeoutError("Function execution timed out")


def compute_upper_bound(sequence: list[float]) -> float:
    """
    Returns the upper bound for a sequence of coefficients.
    Logic taken from Code 1.
    """
    seq_arr = np.array(sequence)
    # Convolve sequence with (1 - sequence)
    convolution_values = np.correlate(seq_arr, 1 - seq_arr, mode="full")
    # Max overlap normalized
    return np.max(convolution_values) / len(sequence) * 2


def verification(sequence: np.ndarray):
    """
    Verify the correctness of the generated sequence based on Code 1 constraints.
    Returns: (is_valid, message)
    """
    # 1. Check that all values are between 0 and 1.
    if not np.all((sequence >= 0) & (sequence <= 1)):
        return False, "All values in the sequence must be between 0 and 1."

    # 2. Check that the sum of values in the sequence is exactly n / 2.0.
    # Note: Using np.isclose matching Code 1 logic
    target_val = len(sequence) / 2.0
    current_sum = np.sum(sequence)

    if not np.isclose(current_sum, target_val, rtol=1e-6):
        return False, (
            f"The sum of values must be exactly n / 2.0. "
            f"Got {current_sum}, expected {target_val}."
        )

    return True, "Valid"


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
    start_time = time.time()
    status = "success"

    try:
        # 1. --- Execution Phase ---
        # Run code using the new dynamic import method with timeout
        # Note: Code 1 uses an adaptive loop that can take time, so we allow a generous timeout
        best_half_seq = run_external_function(program_path, "generate_erdos_data", timeout_seconds=3600)
        eval_time = time.time() - start_time

        # Ensure it is a numpy array
        if not isinstance(best_half_seq, np.ndarray):
            best_half_seq = np.array(best_half_seq)

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

    # 2. --- Reconstruction ---
    # Reconstruct final sequence as per Code 1's main block logic
    try:
        reversed_sequence = best_half_seq[::-1]
        final_sequence = np.concatenate((best_half_seq[:-1], reversed_sequence))
    except Exception as e:
        summary = f"Data reconstruction failed (concatenation): {str(e)}"
        print(summary, file=sys.stderr)
        return {
            "status": "validation_failed",
            "summary": summary,
            "score": 0.0,
            "metrics": {"validity": 0.0},
            "artifacts": {"reason": "Reconstruction Error"},
        }

    # 3. --- Validation Phase ---
    is_valid, message = verification(final_sequence)

    if not is_valid:
        summary = f"Validation failed: {message}"
        print(summary, file=sys.stderr)
        return {
            "status": "validation_failed",
            "summary": summary,
            "score": 0.0,
            "metrics": {"validity": 0.0},
            "artifacts": {"reason": message},
        }

    # 4. --- Metric Computation ---
    # Calculate the overlap (Goal: Minimize this)
    overlap_value = compute_upper_bound(final_sequence)

    # 5. --- Scoring ---
    # Target is 0.380927. Smaller overlap is better.
    # Score = Target / Actual.
    # If Actual < Target, Score > 1.0 (Good).
    # If Actual > Target, Score < 1.0 (Bad).
    if overlap_value <= 0:
        # Avoid division by zero or negative weirdness, though overlap shouldn't be <= 0 ideally
        score = 0.0
    else:
        score = TARGET_VALUE / overlap_value

    summary = (
        f"Success: Valid sequence found. Length: {len(final_sequence)}. "
        f"Upper Bound(less is better): {overlap_value:.8f} (Target: {TARGET_VALUE}). Score: {score:.8f}"
    )
    print(summary)

    return {
        "status": status,
        "summary": summary,
        "score": float(score),
        "metrics": {
            "upper bound": float(overlap_value),
            "target_upper_bound": float(TARGET_VALUE),
            "validity": 1.0,
            "eval_time": float(eval_time),
        },
        "artifacts": {
            "execution_time": f"{eval_time:.2f}s",
            "full_sequence_length": len(final_sequence),
            "best_half_seq": best_half_seq.tolist(),
        },
    }
