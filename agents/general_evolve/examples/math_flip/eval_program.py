"""Evaluation script for math flip puzzles."""

import importlib
import os
import re
import signal
import sys
import time
import traceback


class TimeoutError(Exception):
    """Custom exception raised when a function times out"""

    pass


def timeout_handler(signum, frame):
    """Handle timeout signal"""
    raise TimeoutError("Function execution timed out")


def run_with_timeout(program_path: str, timeout_seconds: int = 20):
    """
    Run the solver program using dynamic import mechanism (no subprocess).
    Calls the run() function and returns its result.
    """

    abs_program_path = os.path.abspath(program_path)
    program_dir, file_name = os.path.split(abs_program_path)
    module_name, _ = os.path.splitext(file_name)

    if not module_name.isidentifier():
        pass

    if program_dir not in sys.path:
        sys.path.insert(0, program_dir)

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout_seconds)

    try:
        if module_name in sys.modules:
            program_module = importlib.reload(sys.modules[module_name])
        else:
            program_module = importlib.import_module(module_name)

        func_name = "run"
        if not hasattr(program_module, func_name):
            raise AttributeError(f"Function '{func_name}' not found in {program_path}")

        solutions = getattr(program_module, func_name)()

        return solutions

    except TimeoutError:
        raise TimeoutError(f"Process timed out after {timeout_seconds} seconds.")
    except Exception as e:
        raise RuntimeError(
            f"Program execution failed: {str(e)}\n{traceback.format_exc()}"
        )

    finally:
        signal.alarm(0)

        if program_dir in sys.path:
            sys.path.remove(program_dir)

        if module_name in sys.modules:
            del sys.modules[module_name]


def verify_solution(solution_str: str):
    """
    Verifies a single solution string against all puzzle rules.

    Args:
        solution_str: The formatted equation string (e.g., "81 + 3 = 4 * 9").

    Returns:
        tuple[bool, str | None]: A tuple containing a boolean (True if valid)
                                 and a reason string (if invalid).
    """
    reason = ""
    all_digits_str = re.findall(r"\d", solution_str)
    if len(set(all_digits_str)) != len(all_digits_str):
        reason = f"Validation Error: Digits are not unique in '{solution_str}'."
        return False, reason

    equation_to_eval = solution_str.replace("=", "==")
    try:
        if not eval(equation_to_eval):
            reason = f"Validation Error: Equation '{solution_str}' is mathematically incorrect."
            return False, reason
    except Exception as e:
        return False, f"Error evaluating original equation '{solution_str}': {e}"

    # Define a mapping from digits to their upside-down counterparts
    UPSIDE_DOWN_MAP = {
        "0": "0",
        "1": "1",
        "2": "3",
        "3": "2",
        "4": "7",
        "5": "5",
        "6": "9",
        "7": "4",
        "8": "8",
        "9": "6",
    }

    # Check if all digits can be flipped
    for digit in all_digits_str:
        if digit not in UPSIDE_DOWN_MAP:
            reason = f"Validation Error: Digit '{digit}' in '{solution_str}' is not flippable."
            return False, reason

    parts = re.split(r"([+\-*/=])", solution_str.replace(" ", ""))

    ud_parts = [
        "".join(UPSIDE_DOWN_MAP[d] for d in p)[::-1] if p.isdigit() else p
        for p in parts
    ]

    ud_equation_to_eval = "".join(ud_parts[::-1]).replace("=", "==")

    print(f"Original: {solution_str}")
    print(f"Upside-down to evaluate: {ud_equation_to_eval}")

    try:
        if not eval(ud_equation_to_eval):
            reason = f"Validation Error: Upside-down eq '{ud_equation_to_eval}' from '{solution_str}' is incorrect."
            return False, reason
    except Exception as e:
        return (
            False,
            f"Error evaluating upside-down equation '{ud_equation_to_eval}': {e}",
        )

    return True, None


def evaluate(program_path: str):
    """
    Evaluate the program by running it in a sandbox and checking its output.

    Args:
        program_path: Path to the solver program file.

    Returns:
        Dictionary of metrics including the score, matching the new schema.
    """
    start_time = time.time()
    try:
        # Run the solver with a timeout and get the results
        solutions = run_with_timeout(program_path, timeout_seconds=1800)
        eval_time = time.time() - start_time

        is_correct = True
        failed_reason = f"Run result from program: {solutions}\n"
        p1_failed = False
        p2_failed = False
        failed_score = 0.6
        success_puzzle = ""

        # Validate Puzzle 1
        p1_sol = solutions.get("puzzle1")
        if not p1_sol:
            print("No solution found for Puzzle 1.", file=sys.stderr)
            failed_reason += "No solution found for Puzzle 1.\n"
            is_correct = False
            failed_score -= 0.3
        else:
            for s in p1_sol:
                is_valid, reason = verify_solution(s)
                if is_valid:
                    failed_score += 0.2
                    is_correct = True
                    success_puzzle += "Puzzle 1: " + s + "\n"
                    break
                if not is_valid:
                    failed_reason += f"Puzzle 1 solution '{s}' is incorrect: {reason}\n"
                    is_correct = False
                    p1_failed = True

        # Validate Puzzle 2
        p2_sol = solutions.get("puzzle2")
        if not p2_sol:
            print("No solution found for Puzzle 2.", file=sys.stderr)
            failed_reason += "No solution found for Puzzle 2.\n"
            is_correct = False
            failed_score -= 0.3
        else:
            for s in p2_sol:
                is_valid, reason = verify_solution(s)
                if is_valid:
                    failed_score += 0.2
                    is_correct = True
                    success_puzzle += "Puzzle 2: " + s + "\n"
                    break
                if not is_valid:
                    failed_reason += f"Puzzle 2 solution '{s}' is incorrect: {reason}\n"
                    is_correct = False
                    p2_failed = True

        if is_correct:
            summary = "Validation Successful!\n"
            summary += f"Puzzle 1: {solutions['puzzle1'][0]}\n"
            summary += f"Puzzle 2: {solutions['puzzle2'][0]}"
            print(summary)

            return {
                "status": "success",
                "summary": summary,
                "score": 1.0,
                "metrics": {
                    "validity": 1.0,
                    "eval_time": float(eval_time),
                    "success_details": success_puzzle,
                },
                "artifacts": {
                    "puzzle1_solution": solutions["puzzle1"][0],
                    "puzzle2_solution": solutions["puzzle2"][0],
                },
            }

        print("Validation Failed.", file=sys.stderr)
        print(failed_reason, file=sys.stderr)

        if p1_failed and p2_failed:
            failed_score -= 0.5
        elif p1_failed:
            failed_score -= 0.4
        elif p2_failed:
            failed_score -= 0.4

        if failed_score >= 1.0:
            failed_score = 0.8
        elif failed_score < 0.0:
            failed_score = 0.0

        summary = f"Validation Failed. Score: {failed_score:.2f}\n{failed_reason}"

        return {
            "status": "validation_failed",
            "summary": summary,
            "score": float(failed_score),
            "metrics": {
                "validity": 0.0,  # Partial validity reflected in score
                "eval_time": float(eval_time),
            },
            "artifacts": {
                "failed_reason": failed_reason,
                "partial_success": success_puzzle,
            },
        }

    except Exception as e:
        eval_time = time.time() - start_time
        error_msg = f"Program execution failed: {str(e)}"
        print(error_msg, file=sys.stderr)
        traceback.print_exc()

        return {
            "status": "execution_failed",
            "summary": error_msg,
            "score": 0.0,
            "metrics": {"eval_time": float(eval_time)},
            "artifacts": {
                "error_type": type(e).__name__,
                "traceback": traceback.format_exc(),
            },
        }
