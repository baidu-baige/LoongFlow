"""
eval program for moe lb
"""

import json
import os
import pickle
import subprocess
import sys
import tempfile
import time
from typing import Any, Dict

import numpy as np

# --- Constants ---
TARGET_IMBALANCE_RATIO = 1.01
TIMEOUT_SECONDS = 60
N_EVAL_RUNS = 100  # Runs 100 times for evaluation


class TimeoutError(Exception):
    """Timeout Error"""

    pass


def redundant_policy(
    n_logical_experts: int,
    ep_size: int,
    n_redundants_per_rank: int,
    workload_history: np.ndarray,
) -> np.ndarray:
    """
    Based on the workload history, decide which experts need redundant copies,
    """
    n_original_per_rank = n_logical_experts // ep_size
    physical_expert_placement = np.zeros(
        (ep_size, n_original_per_rank + n_redundants_per_rank), dtype=int
    )

    for rank in range(ep_size):
        start_idx = rank * n_original_per_rank
        end_idx = start_idx + n_original_per_rank
        physical_expert_placement[rank, :n_original_per_rank] = np.arange(
            start_idx, end_idx
        )

    for current_rank in range(ep_size):
        current_experts = physical_expert_placement[current_rank, :n_original_per_rank]
        expert_workloads = workload_history[current_experts]

        sorted_indices = np.argsort(expert_workloads)[::-1]
        top_indices = sorted_indices[:n_redundants_per_rank]

        for i, idx in enumerate(top_indices):
            expert_id = current_experts[idx]
            target_rank = (current_rank + i + 1) % ep_size
            physical_expert_placement[target_rank, n_original_per_rank + i] = expert_id

    return physical_expert_placement


def run_in_sandbox(
    program_path: str, input_data: Dict[str, np.ndarray]
) -> Dict[str, Any]:
    """
    Run the function to be evaluated in a sandbox process, passing input and output via files.
    """
    with tempfile.NamedTemporaryFile(suffix=".in", delete=False) as infile:
        input_path = infile.name
    output_path = input_path.replace(".in", ".out")

    with open(input_path, "wb") as f:
        pickle.dump(input_data, f)

    script = f"""
import sys, os, pickle, traceback, numpy as np
sys.path.insert(0, os.path.dirname('{program_path}'))
try:
    spec = __import__('importlib.util').util.spec_from_file_location("program", '{program_path}')
    program = __import__('importlib.util').util.module_from_spec(spec)
    spec.loader.exec_module(program)
    with open('{input_path}', 'rb') as f:
        inputs = pickle.load(f)
    allocation_matrix, minimized_max_load = program.solve_lplb_policy(
        inputs['initial_workloads'],
        inputs['physical_expert_placement']
    )
    results = {{
        'allocation_matrix': allocation_matrix,
        'minimized_max_load': minimized_max_load,
        'status': 'success'
    }}
    with open('{output_path}', 'wb') as f:
        pickle.dump(results, f)
except Exception as e:
    with open('{output_path}', 'wb') as f:
        pickle.dump({{'status': 'error', 'error': str(e), 'traceback': traceback.format_exc()}}, f)
"""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False
    ) as script_file:
        script_file.write(script)
        script_path = script_file.name

    try:
        process = subprocess.Popen(
            [sys.executable, script_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        stdout, stderr = process.communicate(timeout=TIMEOUT_SECONDS)
        if process.returncode != 0:
            raise RuntimeError(
                f"Sandbox exited with code {process.returncode}\\n{stdout.decode()}\\n{stderr.decode()}"
            )
        if os.path.exists(output_path):
            with open(output_path, "rb") as f:
                results = pickle.load(f)
            if results.get("status") == "error":
                raise RuntimeError(
                    f"Sandbox execution failed: {results['error']}\\n{results['traceback']}"
                )
            return results
        else:
            raise RuntimeError("Sandbox did not produce output file.")
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
        raise TimeoutError(f"Timeout after {TIMEOUT_SECONDS}s")
    finally:
        for p in [input_path, output_path, script_path]:
            if os.path.exists(p):
                os.unlink(p)


def run_single_evaluation(program_path: str) -> Dict[str, Any]:
    """
    Execute a single evaluation of the program and return the results.
    """
    np.random.seed()
    n_logical_experts, ep_size, n_redundants_per_rank = 256, 16, 4
    workload_history = np.random.rand(n_logical_experts) * 2**20
    placement = redundant_policy(
        n_logical_experts, ep_size, n_redundants_per_rank, workload_history
    )
    initial_workloads = np.random.rand(n_logical_experts) * 2**12
    input_data = {
        "initial_workloads": initial_workloads,
        "physical_expert_placement": placement,
    }

    results = run_in_sandbox(program_path, input_data)
    achieved_max_load = results["minimized_max_load"]
    allocation_matrix = results["allocation_matrix"]

    if not isinstance(allocation_matrix, np.ndarray) or not isinstance(
        achieved_max_load, float
    ):
        raise TypeError("Invalid return types from sandbox.")

    sum_alloc = np.sum(allocation_matrix, axis=1)
    if not np.allclose(sum_alloc, initial_workloads):
        raise ValueError("Flow conservation constraint failed.")
    if not np.all(allocation_matrix >= -1e-6):
        raise ValueError("Non-negativity constraint failed.")

    flat_placement = placement.flatten()
    valid_mask = np.arange(n_logical_experts)[:, np.newaxis] == flat_placement
    if not np.allclose(allocation_matrix[~valid_mask], 0):
        raise ValueError("Placement constraint failed.")

    ep_size_p, n_slots_p = placement.shape
    slot_loads = np.sum(allocation_matrix, axis=0)
    gpu_loads = slot_loads.reshape((ep_size_p, n_slots_p)).sum(axis=1)
    verified_max_load = np.max(gpu_loads) if gpu_loads.size > 0 else 0.0

    total_load = np.sum(initial_workloads)
    avg_load = total_load / ep_size
    imbalance_ratio = verified_max_load / avg_load

    score = min(
        1.0, TARGET_IMBALANCE_RATIO / imbalance_ratio if imbalance_ratio > 0 else 0.0
    )

    return {
        "score": score,
        "imbalance_ratio": imbalance_ratio,
        "verified_max_load": verified_max_load,
    }


def evaluate(program_path: str) -> Dict[str, Any]:
    """
    Evaluate the specified script N_EVAL_RUNS times and return an aggregated structured report.
    """
    start_time = time.time()
    scores, imbalance_ratios, max_loads = [], [], []

    for i in range(N_EVAL_RUNS):
        try:
            single_run_report = run_single_evaluation(program_path)
            scores.append(single_run_report["score"])
            imbalance_ratios.append(single_run_report["imbalance_ratio"])
            max_loads.append(single_run_report["verified_max_load"])
        except (TimeoutError, RuntimeError, TypeError, ValueError) as e:
            eval_time = time.time() - start_time
            return {
                "score": 0.0,
                "status": "execution_failed",
                "summary": f"Evaluation failed on run {i + 1}/{N_EVAL_RUNS}: {str(e)}",
                "metrics": {"eval_time": eval_time, "completed_runs": i},
                "artifacts": {
                    "failure_stage": "single_run_execution",
                    "error_message": str(e),
                },
            }

    eval_time = time.time() - start_time
    mean_score = np.mean(scores)

    return {
        "score": mean_score,
        "status": "success",
        "summary": f"Evaluation successful across {N_EVAL_RUNS} runs. "
        f"Mean score: {mean_score:.4f}, Mean imbalance: {np.mean(imbalance_ratios):.4f}",
        "metrics": {
            "mean_score": mean_score,
            "std_dev_score": np.std(scores),
            "min_score": np.min(scores),
            "max_score": np.max(scores),
            "mean_imbalance_ratio": np.mean(imbalance_ratios),
            "std_dev_imbalance_ratio": np.std(imbalance_ratios),
            "eval_time": eval_time,
            "completed_runs": N_EVAL_RUNS,
        },
        "artifacts": {"test_case_name": f"Dynamic_Avg_{N_EVAL_RUNS}_Runs"},
    }


if __name__ == "__main__":
    program_file = "initial_program.py"
    if not os.path.exists(program_file):
        print(f"Error: File not found at {program_file}")
    else:
        print(f"--- Evaluating {program_file} ({N_EVAL_RUNS} runs) ---")
        report = evaluate(program_file)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        print(f"\nFinal Mean Score: {report.get('score', 0.0):.4f}")
