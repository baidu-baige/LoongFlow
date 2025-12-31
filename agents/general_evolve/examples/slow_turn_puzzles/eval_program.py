"""
Evaluator for slow puzzle probelm (6x6)
Enhanced with artifacts to demonstrate execution feedback
"""

import importlib.util
import os
import sys
import time
import traceback

grid = [
    ["", "", "G", "", "", ""],
    ["R", "", "", "G", "", ""],
    ["", "", "", "", "", ""],
    ["", "", "", "", "", ""],
    ["", "", "", "", "", ""],
    ["", "R", "", "", "", ""],
]


class TimeoutError(Exception):
    """Raised when a timeout occurs."""

    pass


def timeout_handler(signum, frame):
    """Handle timeout signal"""
    raise TimeoutError("Function execution timed out")


def validate_solution(grid, path) -> (bool, str):
    """
    Validate the solution path for the slow puzzle problem.
    """
    if path is None:
        return False, "Path is empty"

    if path[0] != path[-1]:
        return False, "Path is not a cycle: start and end points are different"

    visited = set()
    for point in path[:-1]:
        if point in visited:
            return False, f"Point {point} Repeated visits"
        visited.add(point)

    for i in range(len(path) - 1):
        x1, y1 = path[i]
        x2, y2 = path[i + 1]
        dx = x2 - x1
        dy = y2 - y1
        if abs(dx) > 1 or abs(dy) > 1 or (dx == 0 and dy == 0):
            return False, f"Move from {path[i]} to {path[i + 1]} which is not adjacent"

    n = len(path)
    for i in range(n):
        prev_point = path[i - 1] if i > 0 else path[-2]
        current_point = path[i]
        next_point = path[i + 1] if i < n - 1 else path[1]

        dx_in = current_point[0] - prev_point[0]
        dy_in = current_point[1] - prev_point[1]
        dx_out = next_point[0] - current_point[0]
        dy_out = next_point[1] - current_point[1]

        dot = dx_in * dx_out + dy_in * dy_out
        norm_in_sq = dx_in**2 + dy_in**2
        norm_out_sq = dx_out**2 + dy_out**2

        if dot < 0:
            return False, (
                f"The turning angle from {prev_point} -> {current_point}  "
                + f"to {current_point} -> {next_point} is greater than 90 degrees"
            )

        if dot**2 * 2 < norm_in_sq * norm_out_sq:
            return False, (
                f"The turning angle from {prev_point} -> {current_point}  "
                + f"to {current_point} -> {next_point} exceeds 45 degrees"
            )

    green_squares = [
        (i, j)
        for i in range(len(grid))
        for j in range(len(grid[0]))
        if grid[i][j] == "G"
    ]
    path_set = set(path)
    for green in green_squares:
        if green not in path_set:
            return False, f"Green square {green} is not in the path"

    red_squares = [
        (i, j)
        for i in range(len(grid))
        for j in range(len(grid[0]))
        if grid[i][j] == "R"
    ]
    for red in red_squares:
        if red in path_set:
            return False, f"Red square {red} is in the path"

    return True, f"Validation pass"


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

        if not hasattr(program_module, "solve_puzzle"):
            raise AttributeError(f"Function 'solve_puzzle' not found in {program_path}")

        solution, time_taken = program_module.solve_puzzle(grid)

        print(f"solve_puzzle() returned successfully")
        return solution, time_taken

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
    Evaluate the program by running it once and checking the solution and time_taken

    Args:
        program_path: Path to the program file

    Returns:
        EvaluationResult with metrics and artifacts
    """
    TARGET_VALUE = 1

    try:
        # For constructor-based approaches, a single evaluation is sufficient
        # since the result is deterministic
        start_time = time.time()

        # Use subprocess to run with timeout
        solution, time_taken = run_with_timeout(
            program_path, timeout_seconds=1800  # Single timeout
        )

        end_time = time.time()
        eval_time = end_time - start_time

        # Validate solution
        valid, message = validate_solution(grid, solution)

        if not valid:
            print(f"Solution validation failed: {message}")
            return {
                "status": "execution_failed",
                "score": 0.0,
                "summary": f"Invalid solution: {message}",
                "metrics": {
                    "target_ratio": 0.0,
                },
                "artifacts": {
                    "stderr": message,
                    "failure_stage": "solution_validation",
                    "suggestion": "Please generate valid solution",
                },
            }

        # Target ratio (how close we are to the target)
        target_ratio = valid

        # Combined score - higher is better
        combined_score = target_ratio

        print(
            f"Evaluation: solution={solution}, time_taken={time_taken}s, target={TARGET_VALUE}, "
            + f"ratio={target_ratio:.6f}, time={eval_time:.2f}s"
        )

        # Prepare artifacts with packing details
        artifacts = {
            "time_taken": f"{time_taken:.2f}s",
            "packing_summary": f"solution: {solution}, target_ratio: {target_ratio:.4f}",
            "validation_result": f"{message}",
        }

        # Add successful packing stats for good solutions
        if target_ratio > 0.95:  # Near-optimal solutions
            artifacts["stdout"] = (
                f"Excellent generating! Achieved {target_ratio:.1%} of target value"
            )

        return {
            "status": "success",
            "summary": f"Success: Valid solution found, solution = {solution}, time_taken = {time_taken:.2f}s",
            "score": float(combined_score),
            "metrics": {
                "solution": (solution),
                "target_ratio": float(target_ratio),
                "time_taken": float(time_taken),
            },
            "artifacts": artifacts,
        }

    except TimeoutError as e:
        error_msg = f"Evaluation timed out: {str(e)}"
        print(error_msg)
        return {
            "status": "execution_failed",
            "score": 0.0,
            "summary": f"Execution failed: The program timed out after 1800 seconds.",
            "metrics": {
                "target_ratio": 0.0,
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
            "status": "execution_failed",
            "score": 0.0,
            "summary": f"Execution failed: {str(e)}",
            "metrics": {
                "target_ratio": 0.0,
                "time_taken": 0.0,
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
