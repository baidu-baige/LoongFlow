# Minimum Overlap Problem Example

This example demonstrates how to use the **Evolux** framework to solve a complex optimization problem related to number theory and combinatorics. The goal is to evolve a Python algorithm that constructs a specific sequence of coefficients to minimize their overlap with their complement.

## Problem Description

The objective is to find a sequence of coefficients (relaxed to $[0, 1]$) such that the maximum overlap of the sequence with its complement is minimized, thereby lowering the upper bound of the Erdős minimum overlap constant $C_5$.

For the detailed mathematical definition and problem context, please refer to the official AlphaEvolve problem description:
**[Problem 5: Minimum Overlap Problem](https://google-deepmind.github.io/alphaevolve_repository_of_problems/problems/5.html)**

In this specific task, we aim to find a configuration that yields an upper bound lower than the known bound of **0.380927**.

## Project Structure

*   **`initial_program.py`**: The starting seed code. It implements an adaptive random hill-climbing algorithm that progressively upsamples the sequence resolution to refine the solution.
*   **`eval_program.py`**: The evaluation logic. It executes the generated code, verifies constraints (values in $[0, 1]$, specific sum, symmetry), and calculates the overlap score.
*   **`task_config.yaml`**: The main configuration file defining the LLM prompt, evolution parameters, and the agent components (Planner, Executor, Summarizer).

## How to Run

To start the evolution process, you need to use the `general_evolve_agent.py` entry point. Ensure your `PYTHONPATH` includes the project root so that python can find the `agents` and `evolux` modules.

### 1. Prerequisites

Ensure you are in the root directory of your local project (the directory containing `agents/` and `evolux/`).

### 2. Execution Command

Run the following command to kick off the evolution. This command loads the base configuration and injects the initial code and evaluation logic from the respective files.

```bash
python agents/general_evolve/general_evolve_agent.py \
  --config agents/general_evolve/examples/minimum_overlap_problem/task_config.yaml \
  --initial-file agents/general_evolve/examples/minimum_overlap_problem/initial_program.py \
  --eval-file agents/general_evolve/examples/minimum_overlap_problem/eval_program.py \
  --log-level INFO
```

**Arguments Explanation:**

*   `--config`: Path to the YAML configuration file (`task_config.yaml`).
*   `--initial-file`: Path to the Python file containing the seed code (`initial_program.py`). The content of this file will be injected into `evolve.initial_code`.
*   `--eval-file`: Path to the Python file containing the evaluation logic (`eval_program.py`). The content will be injected into `evolve.evaluator.evaluate_code`.
*   `--log-level`: Sets the logging verbosity (e.g., INFO, DEBUG).

### 3. Configuration Highlights

The `task_config.yaml` is pre-configured with the following strategies:

*   **Planner**: `evolve_planner` (Handles the strategic direction of code modification).
*   **Executor**: `evolve_executor_fuse` (A powerful executor that fuses multiple thought processes/candidates).
*   **Summarizer**: `evolve_summary` (Summarizes the results of the execution for the next iteration).
*   **Target**: The evolution aims to minimize the overlap value below `0.380927`.

## Evolution Process & Results

The system iterates through generations of code, attempting to minimize the convolution overlap.

### Final Result

The Evolux framework successfully identified a sequence configuration that surpasses both the previous State-of-the-Art (SOTA) and the results reported by AlphaEvolve.

**Result Metrics:**
*   **Target Upper Bound:** 0.380927
*   **AlphaEvolve Best:** 0.380924
*   **Evolux Best Result:** **0.3809137564083654**

This result represents a significant improvement in tightening the upper bound for the Erdős minimum overlap problem.

## Troubleshooting

*   **TimeoutError**: The `eval_program.py` enforces a strict timeout (default 3600s for the adaptive algorithm). If the generated code enters an infinite loop or the adaptive strategy is too slow, it will be terminated.
*   **ModuleNotFoundError**: Ensure your `PYTHONPATH` is set correctly. You may need to run `export PYTHONPATH=$PYTHONPATH:.` in the project root before running the command.
```