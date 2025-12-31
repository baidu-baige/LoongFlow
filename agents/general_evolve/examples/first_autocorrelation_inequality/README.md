# First Autocorrelation Inequality

This example demonstrates how to use the **LoongFlow** framework to solve a mathematical optimization problem related to signal processing and inequality theory. The goal is to evolve a Python algorithm that finds a sequence of coefficients to minimize a specific autocorrelation-based evaluation function.

## Problem Description

The objective is to find a sequence of non-negative heights (coefficients) for a step function that minimizes a specific upper bound. The current state-of-the-art (SOTA) benchmark is approximately **1.5098**.

For the problem context and mathematical background, please refer to the official repository:
**[Autocorrelation Problems](https://github.com/google-deepmind/alphaevolve_repository_of_problems/blob/main/experiments/autocorrelation_problems/autocorrelation_problems.ipynb)**

In this specific task, the algorithm attempts to generate a sequence $a$ such that the value $2 \cdot n \cdot \max(b) / (\sum a)^2$ is minimized (where $b$ is the autocorrelation of $a$).

## Project Structure

- **`initial_program.py`**: The starting seed code. It contains the function `run_search_for_best_sequence` and a baseline implementation using Linear Programming (scipy.optimize) to suggest directions for improvement.
- **`eval_program.py`**: The evaluation logic. It executes the generated code with strict timeouts, validates the sequence (non-negative, numeric, non-empty), and calculates the score relative to the target benchmark (1.5053).
- **`task_config.yaml`**: The main configuration file defining the LLM prompt, evolution parameters, and the agent components (Planner, Executor, Summarizer).

## How to Run

To start the evolution process, you need to use the `general_evolve_agent.py` entry point. Ensure your `PYTHONPATH` includes the project root so that python can find the `agents` and `evolux` modules.

### 1. Prerequisites

Ensure you are in the root directory of your local project (the directory containing `agents/` and `evolux/`).

### 2. Execution Command

Run the following command to kick off the evolution. This command loads the base configuration and injects the initial code and evaluation logic from the respective files.

```bash
python agents/general_evolve/general_evolve_agent.py \
  --config agents/general_evolve/examples/first_autocorrelation_inequality/task_config.yaml \
  --initial-file agents/general_evolve/examples/first_autocorrelation_inequality/initial_program.py \
  --eval-file agents/general_evolve/examples/first_autocorrelation_inequality/eval_program.py \
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
- **Target**: The evolution aims for a target score of `1.0` (which implies finding a solution that matches or exceeds the SOTA ratio).

## Evolution Process & Results

The system iterates through generations of code, attempting to minimize the calculated upper bound.

**Result:**

- C1 <= 1.509527314861778.

## Troubleshooting

- **TimeoutError**: The `eval_program.py` enforces a strict timeout (1000s). If the generated sequence search takes too long, it will be terminated.
- **Validation Failed**: If the generated sequence contains negative numbers, NaNs, or sums to near zero, the evaluation will return a failure status.
- **ModuleNotFoundError**: Ensure your `PYTHONPATH` is set correctly. You may need to run `export PYTHONPATH=$PYTHONPATH:.` in the project root before running the command.
