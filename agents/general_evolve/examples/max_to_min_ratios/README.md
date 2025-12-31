# Max to Min Ratios Example

This example demonstrates how to use the **LoongFlow** framework to solve a computational geometry optimization problem. The goal is to evolve a Python algorithm that finds a specific configuration of points in Euclidean space.

## Problem Description

The objective is to find $n$ points in $d$-dimensional Euclidean space that minimize the ratio $R = D_{max} / D_{min}$.

For the detailed mathematical definition and problem context, please refer to the official AlphaEvolve problem description:
**[Problem 50: Max to Min Ratios](https://google-deepmind.github.io/alphaevolve_repository_of_problems/problems/50.html)**

In this specific configuration ($n=16, d=2$), we aim to minimize the squared ratio $R^2$.

## Project Structure

- **`initial_program.py`**: The starting seed code. It contains a basic function signature `optimize_construct` and a naive implementation (placeholder) that needs to be evolved.
- **`eval_program.py`**: The evaluation logic. It executes the generated code in a secure/isolated manner, verifies geometric constraints (distinct points, real numbers), and calculates the score based on the target ratio.
- **`task_config.yaml`**: The main configuration file defining the LLM prompt, evolution parameters (iterations, target score), and the agent components (Planner, Executor, Summarizer).

## How to Run

To start the evolution process, you need to use the `general_evolve_agent.py` entry point. Ensure your `PYTHONPATH` includes the project root so that python can find the `agents` and `evolux` modules.

### 1. Prerequisites

Ensure you are in the root directory of your local project (the directory containing `agents/` and `evolux/`).

### 2. Execution Command

Run the following command to kick off the evolution. This command loads the base configuration and injects the initial code and evaluation logic from the respective files.

```bash
python agents/general_evolve/general_evolve_agent.py \
  --config agents/general_evolve/examples/max_to_min_ratios/task_config.yaml \
  --initial-file agents/general_evolve/examples/max_to_min_ratios/initial_program.py \
  --eval-file agents/general_evolve/examples/max_to_min_ratios/eval_program.py \
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
- **Target**: The evolution aims for a target score of `1.0` (which represents the optimal or best-known ratio).

## Evolution Process & Results

The system iterates through generations of code, attempting to minimize the ratio $D_{max} / D_{min}$. Below are the visualizations of the generated point configurations at different stages of the evolution.

### Final Result

The best solution found by LoongFlow.

![Best Result Visualization](./best_result.png)

**Result Metrics:**

- **Ratio of max distance to min distance::** sqrt(12.88924364235638)

## Troubleshooting

- **TimeoutError**: The `eval_program.py` enforces a strict timeout (default 3600s in config, though internal function calls have shorter timeouts). If the generated code enters an infinite loop, it will be terminated and marked as a failure.
- **ModuleNotFoundError**: Ensure your `PYTHONPATH` is set correctly. You may need to run `export PYTHONPATH=$PYTHONPATH:.` in the project root before running the command.
