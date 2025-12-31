# Second Autocorrelation Inequality Example

This example demonstrates how to use the **LoongFlow** framework to solve a mathematical optimization problem related to signal processing and functional analysis. The goal is to evolve a Python algorithm that constructs a specific step function to improve the lower bound of the second autocorrelation inequality.

## Problem Description

The objective is to find a step function $f$ (represented by a sequence of heights) that maximizes the constant $C_2$ in the inequality $\|f*f\|_2^2 \le C_2 \|f*f\|_1 \|f*f\|_\infty$.

For the detailed mathematical definition and problem context, please refer to the official AlphaEvolve problem description:
**[Autocorrelation Problems](https://github.com/google-deepmind/alphaevolve_repository_of_problems/blob/main/experiments/autocorrelation_problems/autocorrelation_problems.ipynb)**

In this specific configuration, we focus on a step function with 50 equally-spaced intervals on $[-1/4, 1/4]$ to achieve a lower bound $C_2 \ge 0.8962$.

## Project Structure

- **`initial_program.py`**: The starting seed code. It contains the function signature `optimize_lower_bound` and a baseline implementation (a flat step function) that needs to be evolved.
- **`eval_program.py`**: The evaluation logic. It executes the generated code, verifies constraints (non-negative values, correct length), and calculates the lower bound $C_2$.
- **`task_config.yaml`**: The main configuration file defining the LLM prompt, evolution parameters, and the agent components (Planner, Executor, Summarizer).

## How to Run

To start the evolution process, you need to use the `general_evolve_agent.py` entry point. Ensure your `PYTHONPATH` includes the project root.

### 1. Prerequisites

Ensure you are in the root directory of your local project (the directory containing `agents/` and `evolux/`).

### 2. Execution Command

Run the following command to kick off the evolution. This command loads the base configuration and injects the initial code and evaluation logic from the respective files.

```bash
python agents/general_evolve/general_evolve_agent.py \
  --config agents/general_evolve/examples/second_autocorrelation_inequality/task_config.yaml \
  --initial-file agents/general_evolve/examples/second_autocorrelation_inequality/initial_program.py \
  --eval-file agents/general_evolve/examples/second_autocorrelation_inequality/eval_program.py \
  --log-level INFO
```

**Arguments Explanation:**

- `--config`: Path to the YAML configuration file (`task_config.yaml`).
- `--initial-file`: Path to the Python file containing the seed code (`initial_program.py`).
- `--eval-file`: Path to the Python file containing the evaluation logic (`eval_program.py`).
- `--log-level`: Sets the logging verbosity (e.g., INFO, DEBUG).

### 3. Configuration Highlights

The `task_config.yaml` is pre-configured with the following strategies:

- **Planner**: `evolve_planner` (Handles the strategic direction of code modification).
- **Executor**: `evolve_executor_fuse` (Fuses multiple thought processes/candidates to generate robust code).
- **Summarizer**: `evolve_summary` (Summarizes the results for the next iteration).
- **Target**: The evolution aims for a target score of `1.0` (normalized score based on the target $C_2$ value).

## Evolution Process & Results

The system iterates through generations of code, attempting to maximize the lower bound $C_2$.

### Final Result

The evolution process seeks to find a sequence of heights that pushes the value of $C_2$ beyond the known threshold of 0.8962.

**Result Metrics:**

- **Result:** 0.9027021077220739

## Troubleshooting

- **TimeoutError**: The `eval_program.py` enforces a strict timeout (default 3600s in config). If the generated convolution calculation is too inefficient, it will be terminated.
- **ModuleNotFoundError**: Ensure your `PYTHONPATH` is set correctly. You may need to run `export PYTHONPATH=$PYTHONPATH:.` in the project root before running the command.
