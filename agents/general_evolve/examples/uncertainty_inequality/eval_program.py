"""
Optimize Second autocorrelation inequality Problem
"""
import importlib.util
import os
import sys
import signal
import time
import traceback

import numpy as np
import sympy

TIMEOUT = 3600

class TimeoutError(Exception):
    """Raised when a timeout occurs."""
    pass


def timeout_handler(signum, frame):
    """Handle timeout signal"""
    raise TimeoutError("Function execution timed out")

def verify_hermite_combination(coeffs: np.ndarray) -> tuple[sympy.Expr, str]:
    """Computes the Hermite combination for given coefficients."""
    m = len(coeffs)
    rational_coeffs = [sympy.Rational(c) for c in coeffs]
    degrees = np.arange(0, 4 * m + 4, 4)
    x = sympy.symbols('x')
    hps = [
        sympy.polys.orthopolys.hermite_poly(n=i, x=x, polys=False)
        for i in degrees
    ]

    # All but the last coefficient.
    partial_fn: sympy.Expr = sum(
        rational_coeffs[i] * hps[i] for i in range(len(rational_coeffs))
    )

    # Impose the condition that the root at 0 should be 0.
    a = hps[-1].subs(x, 0)
    b = -partial_fn.subs(x, 0)
    last_coeff = b / a
    rational_coeffs.append(sympy.Rational(last_coeff))

    res_fn = sum(rational_coeffs[i] * hps[i] for i in range(len(rational_coeffs)))

    if sympy.limit(res_fn, x, sympy.oo) < 0:
        res_fn = -res_fn

    x = sympy.symbols('x')
    # Check the value at 0 is 0.
    error_msg = ""
    value_at_0 = res_fn.subs(x, 0)
    if value_at_0 != 0:
        error_msg = f'The value at 0 is {value_at_0} != 0.'
    if sympy.limit(res_fn, x, sympy.oo) <= 0:
        error_msg = 'Limit at infty is not positive.'
    return res_fn, error_msg

def get_upper_bound(coeffs: np.ndarray) -> float:
  """Computes the upper bound for the given Hermite combination."""
  g_exp, error_msg = verify_hermite_combination(coeffs)
  if error_msg:
      return 0.0

  x = sympy.symbols('x')
  gq_fn = sympy.exquo(g_exp, x**2)
  rroots = sympy.real_roots(gq_fn, x)
  approx_roots = list()
  largest_sign_change = 0

  for root in rroots:
    approx_root = root.eval_rational(n=200)
    approx_root_p = approx_root + sympy.Rational(1e-198)
    approx_root_m = approx_root - sympy.Rational(1e-198)
    approx_roots.append(approx_root)
    is_sign_change = (
        (gq_fn.subs(x, approx_root_p) > 0 and gq_fn.subs(x, approx_root_m) < 0)
        or (
            gq_fn.subs(x, approx_root_p) < 0
            and gq_fn.subs(x, approx_root_m) > 0
        )
    )
    if is_sign_change:
      largest_sign_change = max(largest_sign_change, approx_root)

  return float(largest_sign_change**2) / (2 * np.pi)


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
    Evaluate the program by running it once and checking

    Args:
        program_path: Path to the program file

    Returns:
        EvaluationResult with metrics and artifacts
    """
    # Target value from the paper
    TARGET_VALUE = 0.3521
    start_time = time.time()

    try:
        # For constructor-based approaches, a single evaluation is sufficient
        # since the result is deterministic
        # Use subprocess to run with timeout
        best_coefficients = run_external_function(
            program_path, "find_coefficients", timeout_seconds=3600
        )

        # Ensure centers and radii are numpy arrays
        if not isinstance(best_coefficients, np.ndarray):
            best_coefficients = np.array(best_coefficients)

        # Validate solution
        final_coefficients, error = verify_hermite_combination(best_coefficients)
        if error != "":
            return {
                "status": "validation_failed",
                "score": 0.0,
                "summary": error,
                "metrics": {"validity": 0.0, "c_upper_bound": 0.0},
                "artifacts": {"reason": "Geometric constraints not met."},
            }

        upper_bound = get_upper_bound(best_coefficients)

        # Target ratio (how close we are to the target)
        target_ratio = TARGET_VALUE / upper_bound if not error else 0.0

        # Validity score
        validity = 1.0 if not error else 0.0

        # Combined score - higher is better
        combined_score = target_ratio * validity

        return {
            "status": "success",
            "summary": f"Success: Valid upper_bound found. C_upper_bound = {upper_bound}, Score: {combined_score:.4f}",
            "score": float(combined_score),
            "metrics": {
                "c_upper_bound": float(upper_bound),
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