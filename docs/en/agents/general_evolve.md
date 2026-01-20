# General Evolve Agent

The General Evolve Agent is a core component of the LoongFlow framework, specifically designed to solve complex mathematical problems, algorithm optimization tasks, and open-domain problem solving. It adopts the **PES (Plan-Execute-Summary)** thinking paradigm, driving the evolution of the agent through structured thinking and continuous learning.

## Overview

The General Evolve Agent combines evolutionary algorithms with reasoning capabilities, implementing several key features:

- **PES Paradigm**: A cyclical thinking process of Planning, Executing, and Summarizing.
- **Multi-Island Evolution Architecture**: Maintains solution diversity to avoid local optima.
- **Structured Memory System**: Accumulates experiential knowledge to support long-term learning.
- **Real-Time Visualization Monitoring**: Provides a complete visual interface for the evolution process.

## Environment Preparation

Ensure you have installed Python 3.12+ and use `uv` for dependency management:

```bash
# Execute in the project root directory
uv sync
```

## Task Configuration

### Configuration File Structure

Each task requires a YAML configuration file. An example structure is as follows:

```yaml
# Global directory configuration
workspace_path: "./output"

# LLM Configuration (supports OpenAI, Gemini, DeepSeek, etc.)
llm_config:
  url: "https://your-llm-api/v1"
  api_key: "your-api-key"
  model: "openai/gemini-3-pro-preview"
  temperature: 0.8
  context_length: 128000
  max_tokens: 32768

# Component Configuration (Planner, Executor, Summarizer)
planners:
  evolve_planner:
    react_max_steps: 10

executors:
  evolve_executor_fuse:
    max_rounds: 3
    react_max_steps: 15
    score_threshold: 0.95

summarizers:
  evolve_summary:
    react_max_steps: 6

# Evolution Process Configuration
evolve:
  task: "Your task description..."
  planner_name: "evolve_planner"
  executor_name: "evolve_executor_fuse"
  summary_name: "evolve_summary"
  max_iterations: 200
  target_score: 1.0
  concurrency: 3
  
  # Evaluator Configuration
  evaluator:
    timeout: 1200
    
  # Database Configuration
  database:
    storage_type: "in_memory"
    num_islands: 3
    population_size: 90
    checkpoint_interval: 1
```

### Code File Preparation

It is recommended to divide task-related code into three files:

#### 1. Initial Code (`initial_program.py`)

Contains the basic implementation framework of the problem, serving as the starting point for the evolutionary process:

```python
# EVOLVE-BLOCK-START
"""Your initial algorithm implementation"""
import numpy as np

def your_initial_solution(problem_parameters):
    # Basic implementation; the evolution process will improve upon this
    return solution

# EVOLVE-BLOCK-END
```

#### 2. Evaluation Code (`eval_program.py`)

Contains the evaluation logic used to judge various solutions during the evolution process:

```python
def evaluate(solution_code_path):
    """
    Evaluation function, returns a dictionary containing score and status info
    """
    try:
        # Execute the solution and evaluate
        result = run_solution(solution_code_path)
        return {
            "status": "success",
            "score": calculated_score,
            "metrics": {"performance": value},
            "artifacts": {"reasoning": "Detailed evaluation results"}
        }
    except Exception as e:
        return {
            "status": "execution_failed",
            "score": 0.0,
            "summary": f"Execution failed: {str(e)}"
        }
```

#### 3. Task Description File

Describe the problem goals and constraints in detail using text. This content can also be written in the `task` field under `evolve` in the configuration file (you can refer to `agents/general_evolve/examples/packing_circle_in_unit_square/task_config.yaml`).

## Running Process

### Start Task

Use the script provided by the project to run the task:

```bash
# Install task-specific dependencies
uv pip install -r ./agents/general_evolve/examples/your_task_name/requirements.txt

# Start the task (run in background)
./run_task.sh packing_circle_in_unit_square --background

# View real-time logs
tail -f ./agents/general_evolve/examples/packing_circle_in_unit_square/run.log

# Stop the task
./run_task.sh stop packing_circle_in_unit_square
```

### Manual Run (For Debugging)

If you need finer control, you can use the Python script directly:

```bash
python agents/general_evolve/general_evolve_agent.py \
  --config agents/general_evolve/examples/your_task_name/task_config.yaml \
  --initial-file agents/general_evolve/examples/your_task_name/initial_program.py \
  --eval-file agents/general_evolve/examples/your_task_name/eval_program.py \
  --max-iterations 500 \
  --log-level INFO
```

### Resume from Checkpoint

If the task is interrupted, you can resume from the latest checkpoint:

```bash
python agents/general_evolve/general_evolve_agent.py \
  --config config.yaml \
  --checkpoint-path ./output/database/checkpoints/checkpoint-checkpoint-iter-89-66
```

## Output Directory Structure

Upon completion, the `output` directory will contain the following structure:

```
output/
├── database/
│   └── checkpoints/
│       └── checkpoint-checkpoint-iter-{iter_num}-{complete_num}/
│           ├── solutions/           # JSON files for all solutions
│           ├── best_solution.json   # The best solution found
│           └── metadata.json        # Metadata (best score, iteration info, etc.)
├── {iteration_number}/
│   ├── planner/                     # Planner phase output
│   │   ├── best_plan.txt           # Best plan
│   │   └── plan_{id}.txt           # Detailed plan
│   ├── executor/                    # Executor phase output
│   │   ├── best_solution.py        # Best solution code
│   │   └── solution_{id}.py        # Generated solution
│   └── summarizer/                  # Summarizer phase output
│       └── best_summary.txt        # Phase summary
└── evaluator/
    └── eval_{UUID}/                 # Evaluation process record
        ├── evaluation_result.json   # Evaluation result
        └── llm_code_{UUID}.py      # The code being evaluated
```

### Output File Descriptions

- **Checkpoint File**: Saves the evolution state, supporting breakpoint resumption.
- **Solution File**: Contains the generated code, score, parent information, etc.
- **Evaluation File**: Detailed evaluation process and results.
- **Log File**: Complete execution logs for debugging.

## Visualization Monitoring

LoongFlow provides a real-time visualization interface to monitor the evolution process:

### Start Visualization Server

```bash
# Execute in the project root directory
python agents/general_evolve/visualizer/visualizer.py \
  --port 8888 \
  --checkpoint-path output/database/checkpoints
```

### Visualization Features

Access `http://localhost:8888` to view the following features:

- **🌳 Evolutionary Tree View**: Displays parent-child relationships of solutions.
- **📈 Score History**: Shows the trend of scores over iterations.
- **🔍 Code Diff**: Compares code modifications between different versions.
- **🗺️ Island Map**: Visualizes the multi-island evolution strategy.
- **⚡ Real-Time Updates**: Automatically refreshes to display the latest evolution state.

### Interface Characteristics

1.  **Solution Tree**: Displays all solutions and their relationships in a tree structure.
2.  **Score Trend Chart**: Shows the best score and average score for each generation.
3.  **Code Diff Viewer**: Highlights code content modifications.
4.  **Filtering and Search**: Filter by score, iteration, island, and other conditions.

## Example Projects

The project provides several examples for reference:

- `packing_circle_in_unit_square` - Circle packing problem.
- `max_to_min_ratios` - Optimization of extreme value ratios.
- `uncertainty_inequality` - Mathematical inequality proof.

Each example contains complete configuration files and code, serving as reference templates for new tasks.

## Troubleshooting

### Common Issues

1.  **Module Import Error**

    ```bash
    # Ensure PYTHONPATH includes the project root directory
    export PYTHONPATH=$PYTHONPATH:.
    ```

2.  **LLM API Configuration Error**
    - Check the URL and API Key in `llm_config`.
    - Confirm the model name format is correct (e.g., `openai/gemini-3-pro-preview`).

3.  **Evaluation Timeout**
    - Check the `evaluator.timeout` setting.
    - Optimize the performance of the evaluation code.

### Debugging Tips

- Use `--log-level DEBUG` to get detailed logs.
- Check the evaluation records in the `output/evaluator/` directory.
- View the visualization interface to understand the evolution state.

## Best Practices

1.  **Task Design**
    - Define clear objective functions and constraints.
    - Provide a reasonable initial solution.
    - Ensure stable evaluation logic.

2.  **Parameter Tuning**
    - Set the number of iterations based on problem complexity.
    - Adjust the number of islands to balance exploration and exploitation.
    - Set reasonable timeout durations.

3.  **Monitoring Optimization**
    - Regularly check the visualization interface.
    - Analyze score trend charts to guide parameter adjustments.
    - Save important checkpoints for future analysis.

By following these guidelines, you can fully utilize the powerful capabilities of the General Evolve Agent to solve complex optimization and algorithm design problems.