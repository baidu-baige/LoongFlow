# LoongFlow Framework - General Evolutionary Agent

## Environment Preparation

Ensure that you have installed Python 3.12+ and the dependency libraries required by the project (such as `pydantic`, `pyyaml`, etc.). It is recommended to use `uv` to manage the environment and install dependencies in the project root directory.

```bash
uv sync
```

## Configuration File

Before running, you need to prepare a YAML configuration file (e.g., `config.yaml`). This file defines the LLM configuration, evolutionary parameters, and settings for various components.

**Example Configuration Structure:**

```yaml
# Global directory configuration
workspace_path: "./output"

# LLM Configuration
llm_config:
  url: "http://your-llm-api/v1"
  api_key: "your-api-key"
  model: "deepseek-r1-250528"
  # ... other parameters

# Component Configuration (Planner, Executor, Summarizer)
planners:
  evolve_planner: { ... }
executors:
  evolve_executor_fuse: { ... }
summarizers:
  evolve_summary: { ... }

# Evolutionary Process Configuration
evolve:
  task: "Find n points in d-dimensional space..."
  planner_name: "evolve_planner"
  executor_name: "evolve_executor_fuse"
  summary_name: "evolve_summary"
  max_iterations: 1000
  target_score: 1.0

  # Evaluator Configuration
  evaluator:
    timeout: 1200

  # Database/Population Configuration
  database:
    storage_type: "in_memory"
    population_size: 100
```

## Usage

The core entry point of the project is `general_evolve_agent.py`. You can flexibly override settings in the configuration file via command-line arguments.

### Command-line Arguments

| Argument            | Required | Default | Description                                                                                |
| :------------------ | :------: | :-----: | :----------------------------------------------------------------------------------------- |
| `-c`, `--config`    | **Yes**  |    -    | Path to the YAML configuration file.                                                       |
| `--checkpoint-path` |    No    |  None   | Specify the Checkpoint directory path to resume the previous evolutionary state.           |
| `--task`            |    No    |  None   | Override the task description text in the configuration file.                              |
| `--task-file`       |    No    |  None   | Read the task description from a file (priority is higher than `--task`).                  |
| `--initial-file`    |    No    |  None   | **Specify the initial code file path**. Overrides `initial_code` in the configuration.     |
| `--eval-file`       |    No    |  None   | **Specify the evaluation code file path**. Overrides `evaluate_code` in the configuration. |
| `--workspace-path`  |    No    |  None   | Override the working directory of the evaluator.                                           |
| `--max-iterations`  |    No    |  None   | Override the maximum number of evolutionary iterations.                                    |
| `--target-score`    |    No    |  None   | Override the target score.                                                                 |
| `--planner`         |    No    |  None   | Specify the Planner component name to use.                                                 |
| `--executor`        |    No    |  None   | Specify the Executor component name to use.                                                |
| `--summary`         |    No    |  None   | Specify the Summary component name to use.                                                 |
| `--log-level`       |    No    |  None   | Set the log level (DEBUG, INFO, WARNING, etc.).                                            |
| `--log-path`        |    No    |  None   | Override the directory where log files are saved.                                          |

### Specifying Task and Code Files

To keep the configuration file clean, it is recommended to store the **task description**, **initial code** (optional), and **evaluation code** (usually mandatory) as separate files and pass them in via command-line arguments.

1.  **Initial Code (`--initial-file`)**: The starting code for population evolution.
2.  **Evaluation Code (`--eval-file`)**: The Python script containing the evaluation logic.

## Complete Running Example

Assuming your file structure is as follows:

- `config.yaml`: Basic configuration file
- `tasks/math_problem.txt`: Specific mathematical task description
- `data/init_script.py`: Initial simple algorithm implementation
- `data/evaluator.py`: Test script used for scoring

You can start the evolutionary process using the following command:

```bash
python general_evolve_agent.py \
    --config config.yaml \
    --task-file tasks/math_problem.txt \
    --initial-file data/init_script.py \
    --eval-file data/evaluator.py \
    --executor evolve_executor_fuse \
    --max-iterations 500 \
    --log-level INFO
```

**Resuming from Checkpoint Example:**

If the task is interrupted, you can continue running by specifying the checkpoint directory:

```bash
python general_evolve_agent.py \
    --config config.yaml \
    --checkpoint-path ./output/database/checkpoints/checkpoint-checkpoint-iter-89-66
```

## Visualization

LoongFlow provides visualization tools to monitor the evolutionary process, score trends, and population status.

Start the visualization service:

```bash
cd visualizer
python visualizer.py --port 8888 --checkpoint-path output/database/checkpoints
```

> **Note**: The `checkpoint-root` parameter is based on the project root directory and automatically appends the subsequent path to locate Checkpoint data.

After startup, please visit `http://localhost:8888` in your browser to view the real-time dashboard.
