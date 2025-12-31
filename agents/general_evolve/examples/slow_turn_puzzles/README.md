# Slow Turn Puzzles Example

This example demonstrates how to use the **LoongFlow** framework to solve a constraint satisfaction puzzle. The goal is to evolve a Python algorithm that finds a specific looping path on a grid subject to strict turning and visitation rules.

## Problem Description

The objective is to find a valid loop on a 6x6 grid that:

1.  Passes through all "Green" (G) squares.
2.  Avoids all "Red" (R) squares.
3.  Makes only "shallow" turns (turns must not exceed 45 degrees relative to the previous direction).
4.  Visits each square at most once (except for the start/end point).

For more puzzles of this type, please refer to the original source:
**[Slow Turn Puzzles](https://erich-friedman.github.io/published/slow/index.html)**

In this specific configuration, we aim to find _any_ valid solution that satisfies all constraints (Score = 1.0).

## Project Structure

- **`initial_program.py`**: The starting seed code. It contains a basic backtracking/DFS implementation (`find_loop`) that attempts to find a path but may not fully respect the "shallow turn" constraints or performance requirements initially.
- **`eval_program.py`**: The evaluation logic. It executes the generated code, validates the path against all geometric and coloring constraints (Green/Red squares, 45-degree turn limit), and ensures the path is a closed loop.
- **`task_config.yaml`**: The main configuration file defining the LLM prompt, evolution parameters, and the agent components. It includes the specific validation logic in the task description to guide the LLM.

## How to Run

To start the evolution process, you need to use the `general_evolve_agent.py` entry point. Ensure your `PYTHONPATH` includes the project root so that python can find the `agents` and `evolux` modules.

### 1. Prerequisites

Ensure you are in the root directory of your local project (the directory containing `agents/` and `evolux/`).

### 2. Execution Command

Run the following command to kick off the evolution. This command loads the base configuration and injects the initial code and evaluation logic from the respective files.

```bash
python agents/general_evolve/general_evolve_agent.py \
  --config agents/general_evolve/examples/slow_turn_puzzles/task_config.yaml \
  --initial-file agents/general_evolve/examples/slow_turn_puzzles/initial_program.py \
  --eval-file agents/general_evolve/examples/slow_turn_puzzles/eval_program.py \
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
- **Target**: The evolution aims for a target score of `1.0` (which implies a fully valid solution has been found).

## Evolution Process & Results

The system iterates through generations of code to fix the logic regarding the specific "135-degree internal angle" (shallow turn) constraint and ensure valid pathfinding performance.

### Final Result

The best solution is a list of coordinates representing the valid loop on the grid.

**Result Metrics:**

- **Status:** Success
- **Target Ratio:** 1.0 (Validation Passed)
- **Time Taken:** Optimized for fast execution.

## Troubleshooting

- **TimeoutError**: The `eval_program.py` enforces a timeout (default 1800s). If the generated backtracking algorithm is too inefficient or enters an infinite loop, it will be terminated.
- **ModuleNotFoundError**: Ensure your `PYTHONPATH` is set correctly. You may need to run `export PYTHONPATH=$PYTHONPATH:.` in the project root before running the command.
