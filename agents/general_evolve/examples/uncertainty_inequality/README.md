# Uncertainty Inequality Example

This example demonstrates how to use the **LoongFlow** framework to solve a complex mathematical optimization problem involving Fourier analysis and Hermite polynomials. The goal is to evolve a Python algorithm that finds specific coefficients to improve the upper bound of a mathematical constant.

## Problem Description

The objective is to find coefficients $c_k$ for a linear combination of Hermite polynomials $P(x)$ that minimize the upper bound for the constant $C_4$ in the uncertainty inequality $A(f)A(\hat{f}) \ge C_4$.

For the detailed mathematical definition and problem context, please refer to the official AlphaEvolve results repository:
**[Uncertainty Inequality Problem](https://github.com/google-deepmind/alphaevolve_results/tree/main)**

In this specific task, we aim to find a configuration of coefficients that yields an upper bound $\le 0.3521$.

## Project Structure

- **`initial_program.py`**: The starting seed code. It contains a basic function signature `find_coefficients` and a helper `verify_hermite_combination` that needs to be utilized by the evolved code.
- **`eval_program.py`**: The evaluation logic. It executes the generated code, validates the polynomial properties (roots, limits), and calculates the upper bound for $C_4$.
- **`task_config.yaml`**: The main configuration file defining the LLM prompt, evolution parameters, and the agent components (Planner, Executor, Summarizer).

## How to Run

To start the evolution process, you need to use the `general_evolve_agent.py` entry point. Ensure your `PYTHONPATH` includes the project root so that python can find the `agents` and `evolux` modules.

### 1. Prerequisites

Ensure you are in the root directory of your local project (the directory containing `agents/` and `evolux/`).

### 2. Execution Command

Run the following command to kick off the evolution. This command loads the base configuration and injects the initial code and evaluation logic from the respective files.

```bash
python agents/general_evolve/general_evolve_agent.py \
  --config agents/general_evolve/examples/uncertainty_inequality/task_config.yaml \
  --initial-file agents/general_evolve/examples/uncertainty_inequality/initial_program.py \
  --eval-file agents/general_evolve/examples/uncertainty_inequality/eval_program.py \
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
- **Target**: The evolution aims for a target score of `1.0` (which corresponds to successfully finding an upper bound $\le 0.3521$).

## Evolution Process & Results

The system iterates through generations of code, searching for optimal coefficients $(c_0, c_1, c_2, ...)$ that define the polynomial $P(x)$.

**Result Metrics:**

- Result: 0.3520991044321593

## Troubleshooting

- **TimeoutError**: The `eval_program.py` enforces a strict timeout (default 3600s). If the generated mathematical search enters an infinite loop or takes too long to converge, it will be terminated.
- **SymPy Errors**: This problem relies heavily on symbolic mathematics. Ensure `sympy` is installed and compatible, as the evaluation involves polynomial roots and limits.
- **ModuleNotFoundError**: Ensure your `PYTHONPATH` is set correctly. You may need to run `export PYTHONPATH=$PYTHONPATH:.` in the project root before running the command.
