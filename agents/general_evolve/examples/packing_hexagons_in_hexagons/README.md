# Packing Hexagons in Hexagons Example

This example demonstrates how to use the **LoongFlow** framework to solve a complex geometric packing optimization problem. The goal is to evolve a Python algorithm that packs 11 unit hexagons into the smallest possible enclosing regular hexagon.

## Problem Description

The objective is to pack 11 non-overlapping unit regular hexagons (side length 1) into a larger enclosing regular hexagon such that the side length of the outer hexagon is minimized.

For the detailed mathematical definition and problem context, please refer to the official AlphaEvolve results repository:
**[AlphaEvolve Results Repository](https://github.com/google-deepmind/alphaevolve_results/tree/main)**

In this specific configuration ($n=11$), we aim to minimize the side length of the outer hexagon, with a target value below 3.931.

## Project Structure

- **`initial_program.py`**: The starting seed code. It contains a basic function signature `optimize_construct` and a heuristic implementation (hybrid optimization with pattern matching and physics simulation) that needs to be evolved.
- **`eval_program.py`**: The evaluation logic. It executes the generated code in a secure/isolated manner, verifies geometric constraints (non-overlapping inner hexagons, containment within the outer hexagon), and calculates the score based on the outer hexagon's side length.
- **`task_config.yaml`**: The main configuration file defining the LLM prompt, evolution parameters (iterations, target score), and the agent components (Planner, Executor, Summarizer).

## How to Run

To start the evolution process, you need to use the `general_evolve_agent.py` entry point. Ensure your `PYTHONPATH` includes the project root so that python can find the `agents` and `evolux` modules.

### 1. Prerequisites

Ensure you are in the root directory of your local project (the directory containing `agents/` and `evolux/`).

### 2. Execution Command

Run the following command to kick off the evolution. This command loads the base configuration and injects the initial code and evaluation logic from the respective files.

```bash
python agents/general_evolve/general_evolve_agent.py \
  --config agents/general_evolve/examples/packing_hexagons_in_hexagons/task_config.yaml \
  --initial-file agents/general_evolve/examples/packing_hexagons_in_hexagons/initial_program.py \
  --eval-file agents/general_evolve/examples/packing_hexagons_in_hexagons/eval_program.py \
  --log-level INFO
```

**Arguments Explanation:**

- `--config`: Path to the YAML configuration file (`task_config.yaml`).
- `--initial-file`: Path to the Python file containing the seed code (`initial_program.py`). The content of this file will be injected into `evolve.initial_code`.
- `--eval-file`: Path to the Python file containing the evaluation logic (`eval_program.py`). The content will be injected into `evolve.evaluator.evaluate_code`.
- `--log-level`: Sets the logging verbosity (e.g., INFO, DEBUG).

### 3. Configuration Highlights

The `task_config.yaml` is pre-configured with the following strategies:

- **Planner**: `algo_planner` (Handles the strategic direction of code modification).
- **Executor**: `algo_executor_fuse` (A powerful executor that fuses multiple thought processes/candidates).
- **Summarizer**: `algo_summary` (Summarizes the results of the execution for the next iteration).
- **Target**: The evolution aims for a target score of `1.0` (calculated as `3.931 / outer_hex_side_length`).

## Evolution Process & Results

The system iterates through generations of code, attempting to minimize the side length of the outer hexagon while ensuring all 11 unit hexagons fit inside without overlapping.

### Final Result

The best solution found by LoongFlow.

**Result Metrics:**

- **Outer Hexagon Side Length:** < 3.929515949646492

## Troubleshooting

- **TimeoutError**: The `eval_program.py` enforces a strict timeout (default 1800s in config). If the generated code enters an infinite loop or the physics simulation takes too long, it will be terminated and marked as a failure.
- **ModuleNotFoundError**: Ensure your `PYTHONPATH` is set correctly. You may need to run `export PYTHONPATH=$PYTHONPATH:.` in the project root before running the command.
