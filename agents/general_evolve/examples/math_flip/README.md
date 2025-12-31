# Math Flip Puzzle Example

This example demonstrates how to use the **LoongFlow** framework to solve a logic puzzle involving digit substitution and geometric transformations. The goal is to evolve a Python algorithm that finds specific assignments of digits to equations such that they remain valid when viewed upside down.

## Problem Description

The objective is to fill in the blank spaces in given equations with digits (0-9) so that the equation is mathematically true, and remains true even when the entire equation is rotated 180 degrees (viewed upside down).

For the problem context and origin, please refer to the US Puzzle Championship 2012 archive:
**[US Puzzle Championship 2012](https://erich-friedman.github.io/puzzle/champ/US2012/)**

In this specific task, we aim to find the exact unique digit configuration that satisfies all constraints for two specific puzzles.

## Project Structure

- **`initial_program.py`**: The starting seed code. It contains a basic function signature `solve_puzzles` and a baseline implementation that attempts to solve the constraint satisfaction problem.
- **`eval_program.py`**: The evaluation logic. It executes the generated code, verifies that the digits are unique, checks the mathematical correctness of the equations, and most importantly, validates the "upside-down" logic using a specific digit mapping.
- **`task_config.yaml`**: The main configuration file defining the LLM prompt, evolution parameters, and the agent components (Planner, Executor, Summarizer).

## How to Run

To start the evolution process, you need to use the `general_evolve_agent.py` entry point. Ensure your `PYTHONPATH` includes the project root so that python can find the `agents` and `evolux` modules.

### 1. Prerequisites

Ensure you are in the root directory of your local project (the directory containing `agents/` and `evolux/`).

### 2. Execution Command

Run the following command to kick off the evolution. This command loads the base configuration and injects the initial code and evaluation logic from the respective files.

```bash
python agents/general_evolve/general_evolve_agent.py \
  --config agents/general_evolve/examples/math_flip/task_config.yaml \
  --initial-file agents/general_evolve/examples/math_flip/initial_program.py \
  --eval-file agents/general_evolve/examples/math_flip/eval_program.py \
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
- **Target**: The evolution aims for a target score of `1.0`, which indicates that both puzzles have been solved correctly and validated.

## Evolution Process & Results

The system iterates through generations of code, attempting to find the correct digit combinations. Unlike optimization problems that converge slowly, this is a "hit or miss" search where the system aims to discover the logic that satisfies the discrete constraints.

### Final Result

The best solution found by LoongFlow.

**Result Metrics:**

- **Status:** Validation Successful
- **Score:** 1.0
- **Puzzle 1 Solution:** Correct equation string found.
- **Puzzle 2 Solution:** Correct equation string found.

## Troubleshooting

- **TimeoutError**: The `eval_program.py` enforces a strict timeout (default 1800s for the specific run, though the overall config might differ). If the generated code enters an infinite loop or an extremely inefficient search, it will be terminated.
- **ModuleNotFoundError**: Ensure your `PYTHONPATH` is set correctly. You may need to run `export PYTHONPATH=$PYTHONPATH:.` in the project root before running the command.
