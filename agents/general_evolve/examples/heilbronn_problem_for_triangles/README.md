# Heilbronn Problem for Triangles Example

This example demonstrates how to use the **LoongFlow** framework to solve a complex computational geometry optimization problem known as the Heilbronn Triangle Problem. The goal is to evolve a Python algorithm that finds a specific configuration of points within an equilateral triangle.

## Problem Description

The objective is to place $n$ points in an equilateral triangle to maximize the minimum area of any triangle formed by three of those points.

For the detailed mathematical definition and problem context, please refer to the official repository:
**[AlphaEvolve Results Repository](https://github.com/google-deepmind/alphaevolve_results/tree/main)**

In this specific configuration ($n=11$), we aim to maximize the minimum triangle area, surpassing the known benchmark.

## Project Structure

- **`initial_program.py`**: The starting seed code. It contains a basic function signature `run_search_point` and a heuristic implementation (using Halton sequences and adaptive hybrid optimization) that needs to be evolved.
- **`eval_program.py`**: The evaluation logic. It executes the generated code in a secure/isolated manner, verifies geometric constraints (points inside the triangle, non-collinear), and calculates the score based on the target area ($0.0365$).
- **`task_config.yaml`**: The main configuration file defining the LLM prompt, evolution parameters (iterations, target score), and the agent components (Planner, Executor, Summarizer).

## How to Run

To start the evolution process, you need to use the `general_evolve_agent.py` entry point. Ensure your `PYTHONPATH` includes the project root so that python can find the `agents` and `evolux` modules.

### 1. Prerequisites

Ensure you are in the root directory of your local project (the directory containing `agents/` and `evolux/`).

### 2. Execution Command

Run the following command to kick off the evolution. This command loads the base configuration and injects the initial code and evaluation logic from the respective files.

```bash
python agents/general_evolve/general_evolve_agent.py \
  --config agents/general_evolve/examples/heilbronn_problem_for_triangles/task_config.yaml \
  --initial-file agents/general_evolve/examples/heilbronn_problem_for_triangles/initial_program.py \
  --eval-file agents/general_evolve/examples/heilbronn_problem_for_triangles/eval_program.py \
  --log-level INFO
```

**Arguments Explanation:**

- `--config`: Path to the YAML configuration file (`task_config.yaml`).
- `--initial-file`: Path to the Python file containing the seed code (`initial_program.py`). The content of this file will be injected into `evolve.initial_code`.
- `--eval-file`: Path to the Python file containing the evaluation logic (`eval_program.py`). The content will be injected into `evolve.evaluator.evaluate_code`.
- `--log-level`: Sets the logging verbosity (e.g., INFO, DEBUG).

### 3. Configuration Highlights

The `task_config.yaml` is pre-configured with the following strategies:

- **Planner**: `evolve_planner` (Handles the strategic direction of code modification).
- **Executor**: `evolve_executor_fuse` (A powerful executor that fuses multiple thought processes/candidates).
- **Summarizer**: `evolve_summary` (Summarizes the results of the execution for the next iteration).
- **Target**: The evolution aims to surpass the SOTA benchmark for $n=11$.

## Evolution Process & Results

The system iterates through generations of code, attempting to maximize the minimum triangle area. Below are the results of the generated point configurations.

### Final Result

The best solution found by LoongFlow.

**Result Metrics:**

- **Minimum Triangle Area:** 0.0365210671221724 (Surpassing the SOTA of 0.036)

## Troubleshooting

- **TimeoutError**: The `eval_program.py` enforces a strict timeout (default 3600s in config). If the generated code enters an infinite loop or takes too long to converge, it will be terminated and marked as a failure.
- **ModuleNotFoundError**: Ensure your `PYTHONPATH` is set correctly. You may need to run `export PYTHONPATH=$PYTHONPATH:.` in the project root before running the command.
