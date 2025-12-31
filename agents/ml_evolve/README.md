# LoongFlow Framework - ML Evolve Agent

A general-purpose machine learning evolution framework that automatically solves ML problems through LLM-driven code
generation and iterative optimization.

> **Note:** This project is under active development. APIs and configurations may change between versions.

## Benchmark Results

We validate our framework on [MLE-Bench](https://github.com/openai/mle-bench), a benchmark of Kaggle-style ML
competitions. Current results (see `examples/mlebench/competitions/`):

| 🥇 Gold | 🥈 Silver | 🥉 Bronze |
|:-------:|:---------:|:---------:|
|   14    |     4     |     2     |

> Full results will be published after completing comprehensive testing across all competitions.

## Prerequisites

- Ubuntu 22.04+ (recommended)
- [Mamba](https://github.com/conda-forge/miniforge) environment manager
- Git LFS (MLE-Bench only): `sudo apt install git-lfs && git lfs install`
- Kaggle API credentials (MLE-Bench only): See [Kaggle API setup](https://github.com/Kaggle/kaggle-api#api-credentials)
  ```bash
  # Download kaggle.json from https://www.kaggle.com/settings/account
  mkdir -p ~/.kaggle && mv kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
  ```

## Quick Start

### Running MLE-Bench Competitions

```bash
# Initialize environment
./run_mlebench.sh init

# Edit configuration (LLM credentials required, other settings optional)
vim agents/ml_evolve/examples/mlebench/task_config.yaml

# Download competition data
./run_mlebench.sh prepare detecting-insults-in-social-commentary

# Run evolution
./run_mlebench.sh run detecting-insults-in-social-commentary --background

# Monitor progress
tail -f output/logs/evolux.log

# Stop when needed
./run_mlebench.sh stop detecting-insults-in-social-commentary
```

### Running Custom ML Tasks

```bash
# Initialize environment
./run_ml.sh init

# Edit configuration (LLM credentials required, other settings optional)
vim agents/ml_evolve/examples/ml_example/task_config.yaml

# Run evolution (ml_example is a demo with Iris classification)
./run_ml.sh run ml_example --background

# Monitor progress
tail -f output/logs/evolux.log

# Stop when needed
./run_ml.sh stop ml_example
```

## Output Structure

After running a task, results are saved in `output/`:

```
output/
├── <task-uuid>/
│   └── <iteration-id>/           # Each evolution iteration
│       ├── planner/              # Task planning and strategy
│       ├── evocoder/             # Generated ML code
│       ├── executor/             # Execution results
│       └── summary/              # Iteration summary and insights
├── logs/                         # Runtime logs
├── database/                     # Checkpoints, solutions, etc
└── evaluate/                     # Evaluation results and scores
```

## Configuration

The `task_config.yaml` controls the evolution process:

```yaml
workspace_path: "./output"

# LLM Configuration (required)
llm_config:
  url: "http://your-llm-api/v1"
  api_key: "your-api-key"
  model: "openai/gemini-3-flash-preview"
  temperature: 0.8
  context_length: 128000
  max_tokens: 32768
  top_p: 1.0

# Component Configurations
planners:
  ml_planner:
    react_max_steps: 10
    evo_coder_timeout: 3600

executors:
  ml_executor:
    react_max_steps: 10
    evo_coder_timeout: 86400

summarizers:
  ml_summary:
    react_max_steps: 10

# Evolution Configuration
evolve:
  planner_name: "ml_planner"
  executor_name: "ml_executor"
  summary_name: "ml_summary"
  max_iterations: 100
  target_score: 1.0
  concurrency: 1

  evaluator:
    timeout: 1800

  database:
    storage_type: "in_memory"
    num_islands: 3
    population_size: 30
    checkpoint_interval: 5
    sampling_weight_power: 1.0
```

> At minimum, configure `llm_config` with your API credentials. Other parameters can be adjusted based on your
> requirements.

## Creating Custom Tasks

Create a task directory under agents/ml_evolve/examples, with the following structure:

```
your_task/
├── task_config.yaml        # Evolution and LLM configuration
├── eval_program.py         # Scoring logic
├── public/
│   ├── description.md      # Task description (visible to agent)
│   ├── train.csv           # Training data
│   ├── test.csv            # Test features
│   └── sample_submission.csv
└── private/
    └── answer.csv          # Ground truth (hidden from agent)
```

| File                    | Purpose                                      |
|:------------------------|:---------------------------------------------|
| `description.md`        | Task requirements and expected output format |
| `train.csv`             | Labeled training data                        |
| `test.csv`              | Unlabeled test data for predictions          |
| `sample_submission.csv` | Expected submission format                   |
| `answer.csv`            | Ground truth for evaluation                  |
| `eval_program.py`       | Scoring logic, returns score between 0.0-1.0 |

The `eval_program.py` should implement:

```python
def evaluate(task_data_path, best_code_path, artifacts):
    """
    Returns:
        dict with keys: status, summary, score (0.0-1.0), metrics, artifacts
    """
```

Run your custom task:

```bash
./run_ml.sh run your_task --background
```

See `examples/ml_example/` for a complete reference.

## Contributing

This project is under rapid development. Feel free to open issues for feedback and suggestions.

