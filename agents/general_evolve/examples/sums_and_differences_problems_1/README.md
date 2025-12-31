# Sums and Differences (Ruzsa's Inequality) Example

This example demonstrates how to use the **LoongFlow** framework to solve a problem in additive combinatorics. The goal is to evolve a Python algorithm that constructs a set of integers $A$ with specific sumset and difference set properties.

## Problem Description

The objective is to find a set of integers $A$ (where $|A| = n$) that maximizes the ratio related to Ruzsa's inequality. Specifically, we want to maximize the score derived from the sizes of the sumset $A+A$ and the difference set $A-A$.

The score is calculated as:
$$ \text{Score} = \frac{\ln(|A+A|/|A|)}{\ln(|A-A|/|A|)} $$

Where:

- $A+A = \{a+b \mid a, b \in A\}$
- $A-A = \{a-b \mid a, b \in A\}$

For the detailed problem context (Note: Link provided points to the repository source):
**[Problem Source Link](https://github.com/google-deepmind/alphaevolve_repository_of_problems/tree/main/experiments/sums_differences_problems)**

In this specific configuration, the system evolves a search algorithm to find the optimal list of integers.

## Project Structure

- **`initial_program.py`**: The starting seed code. It contains a stochastic search function `search_for_best_set` that mutates a list of integers to improve the score.
- **`eval_program.py`**: The evaluation logic. It executes the generated code, validates the output format (Tuple[np.ndarray, str]), ensures elements are integers, and calculates the score using Numba-optimized functions.
- **`task_config.yaml`**: The main configuration file defining the LLM prompt (DeepSeek), evolution parameters (iterations, target score), and the agent components (Planner, Executor, Summarizer).

## How to Run

To start the evolution process, you need to use the `general_evolve_agent.py` entry point. Ensure your `PYTHONPATH` includes the project root so that python can find the `agents` and `evolux` modules.

### 1. Prerequisites

Ensure you are in the root directory of your local project (the directory containing `agents/` and `evolux/`).

### 2. Execution Command

Run the following command to kick off the evolution. This command loads the base configuration and injects the initial code and evaluation logic from the respective files.

```bash
python agents/general_evolve/general_evolve_agent.py \
  --config agents/general_evolve/examples/sums_and_differences_problems_1/task_config.yaml \
  --initial-file agents/general_evolve/examples/sums_and_differences_problems_1/initial_program.py \
  --eval-file agents/general_evolve/examples/sums_and_differences_problems_1/eval_program.py \
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
- **Executor**: `evolve_executor_fuse` (A powerful executor that fuses multiple thought processes/candidates, configured with a high score threshold).
- **Summarizer**: `evolve_summary` (Summarizes the results of the execution for the next iteration).
- **LLM**: Configured to use `deepseek-r1-250528`.

## Evolution Process & Results

The system iterates through generations of code, attempting to maximize the ratio described above.

### Final Result

The best solution found by LoongFlow in the latest run achieved a significant improvement over the baseline.

**Result Metrics:**

- **Best Score Achieved:** `1.103534711409646`
- **Previous Baseline:** `1.059793`

## Troubleshooting

- **TimeoutError**: The `eval_program.py` enforces a strict timeout (default 3600s in config, though internal function calls have shorter timeouts). If the generated code enters an infinite loop, it will be terminated and marked as a failure.
- **ModuleNotFoundError**: Ensure your `PYTHONPATH` is set correctly. You may need to run `export PYTHONPATH=$PYTHONPATH:.` in the project root before running the command.
- **Numba Compilation Overhead**: The first execution might be slightly slower due to JIT compilation of the `get_score_numba` function.
