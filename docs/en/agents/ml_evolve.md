# ML Evolve Agent - Machine Learning Evolutionary Agent

ML Evolve Agent is an agent component within the LoongFlow framework dedicated to machine learning. It adopts the PES (Plan-Execute-Summary) thinking paradigm, automatically building and optimizing machine learning solutions through structured thinking and continuous learning, specifically focusing on Kaggle-style data science competition tasks.

## Overview

ML Evolve Agent combines evolutionary algorithms with machine learning expertise, specifically optimized for data science tasks, featuring the following key characteristics:

- **Full Process Automation**: Complete ML workflow from data exploration and feature engineering to model training.
- **Competition-Grade Evaluation**: Supports standard Kaggle-style evaluation criteria and leaderboards.
- **Specialized Components**: Built-in ML-specific modules such as `evocoder` (code generator) and `evaluator`.
- **Multi-Objective Optimization**: Balances model accuracy, computational efficiency, and generalization capability.
- **Structured Thinking**: Based on the Plan-Execute-Summary (PES) loop.

## Environment Preparation

Ensure Python 3.12+ is installed and use `uv` for dependency management:

```bash
# Execute in the project root directory
uv venv .venv --python 3.12
source .venv/bin/activate
uv pip install -e .
```

### MLE-Bench Specific Preparation

If you need to run MLE-Bench competitions, additional configuration is required:

```bash
# Initialize MLE-Bench environment
./run_mlebench.sh init

# Configure Kaggle API (for data download)
# Download kaggle.json from https://www.kaggle.com/settings/account
mkdir -p ~/.kaggle && mv kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
```

## Task Configuration

### Configuration File Structure

Each machine learning task requires a YAML configuration file:

```yaml
# Global directory configuration
workspace_path: "./output"

# LLM Configuration (Supports OpenAI, Gemini, DeepSeek, etc.)
llm_config:
  url: "https://your-llm-api/v1"
  api_key: "your-api-key"
  model: "openai/gemini-3-flash-preview"
  temperature: 0.8
  context_length: 128000
  max_tokens: 32768
  top_p: 1.0

# Component Configuration (Planner, Executor, Summarizer)
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

# Evolutionary Process Configuration
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

### Task File Structure

Machine learning tasks need to be organized according to a standard structure:

```
your_task/
├── task_config.yaml        # Evolution and LLM configuration
├── eval_program.py         # Scoring logic
├── public/
│   ├── description.md      # Task description (visible to the agent)
│   ├── train.csv           # Training data
│   ├── test.csv            # Test features
│   └── sample_submission.csv # Example submission format
└── private/
    └── answer.csv          # Ground truth (hidden from the agent)
```

#### File Description

| File | Usage |
|------|------|
| `description.md` | Task requirements, data explanation, and expected output format |
| `train.csv` | Labeled training data |
| `test.csv` | Unlabeled test data |
| `sample_submission.csv` | Expected submission format |
| `answer.csv` | Ground truth for evaluation (invisible to the agent) |
| `eval_program.py` | Scoring logic, returns a score between 0.0-1.0 |

### Writing Evaluation Code

`eval_program.py` needs to implement the standard evaluation interface:

```python
def evaluate(task_data_path, best_code_path, artifacts):
    """
    Evaluation function, returns a dictionary containing score and status info.
    
    Args:
        task_data_path: Path to the task data directory
        best_code_path: Path to the best code file
        artifacts: Evaluation process parameters
        
    Returns:
        dict: Contains status, score, metrics, etc.
    """
    try:
        # Execute the solution and evaluate
        result = run_machine_learning_evaluation(best_code_path)
        return {
            "status": "success",
            "score": result["score"],  # 0.0-1.0
            "metrics": {
                "accuracy": result["accuracy"],
                "f1_score": result["f1"]
            },
            "artifacts": {
                "reasoning": "Detailed evaluation result",
                "predictions": result["predictions"]
            }
        }
    except Exception as e:
        return {
            "status": "execution_failed",
            "score": 0.0,
            "summary": f"Evaluation failed: {str(e)}"
        }
```

## Execution Flow

### Running Custom ML Tasks

```bash
# Initialize environment
./run_ml.sh init

# Edit configuration file (configure LLM credentials, etc.)
vim agents/ml_evolve/examples/ml_example/task_config.yaml

# Start task (run in background)
./run_ml.sh run ml_example --background

# View real-time logs
tail -f ./agents/ml_evolve/examples/ml_example/agent.log

# Stop task
./run_ml.sh stop ml_example
```

### Running MLE-Bench Competitions

```bash
# Initialize MLE-Bench environment
./run_mlebench.sh init

# Download competition data
./run_mlebench.sh prepare detecting-insults-in-social-commentary

# Run evolutionary process
./run_mlebench.sh run detecting-insults-in-social-commentary --background

# Monitor progress
tail -f output/logs/evolux.log
```

### Recovering from Checkpoints

If a task is interrupted, you can resume from the latest checkpoint:

```bash
python agents/ml_evolve/ml_evolve.py \
  --config config.yaml \
  --checkpoint-path ./output/database/checkpoints/checkpoint-checkpoint-iter-50-25
```

## Output Directory Structure

Upon completion, the `output` directory will contain the following structure:

```
output/
├── <task-uuid>/                  # Task Unique Identifier
│   └── <iteration-id>/           # Each evolutionary iteration
│       ├── planner/              # Task planning phase
│       │   ├── best_plan.txt     # Best planning strategy
│       │   └── plan_{ID}.txt     # Detailed planning process
│       ├── executor/             # Execution phase
│       │   ├── best_solution.py  # Best machine learning code
│       │   └── solution_{ID}.py  # Generated solutions
│       └── summarizer/           # Summary phase
│           └── best_summary.txt  # Phase summary and improvement suggestions
├── database/                     # Evolutionary state database
│   └── checkpoints/
│       └── checkpoint-checkpoint-iter-{iter}-{ID}/
│           ├── solutions/        # JSON records of all solutions
│           ├── best_solution.json # Information on the best solution
│           └── metadata.json     # Metadata (best score, iteration info, etc.)
├── evaluator/                    # Evaluation process records
│   └── eval_{UUID}/              # Evaluation process record
│       ├── evaluation_result.json # Evaluation result
│       └── llm_code_{UUID}.py   # The evaluated machine learning code
└── logs/                         # Runtime logs
    └── evolux.log
```

### Output File Description

- **Checkpoint Files**: Save evolutionary state, supporting resume capability.
- **Solution Files**: Contain generated ML model code, hyperparameters, and evaluation scores.
- **Evaluation Files**: Detailed model performance evaluation results and metrics.
- **Planning Files**: The agent's thinking strategy and experimental design for the ML task.
- **Summary Files**: Phase learning summaries and directions for future improvements.

## Visualization Monitoring

ML Evolve Agent shares a general visualization system to monitor the evolutionary process:

### Start Visualization Server

```bash
# Execute in the project root directory
python agents/general_evolve/visualizer/visualizer.py \
  --port 8888 \
  --checkpoint-path output/database/checkpoints
```

### Visualization Features

Access `http://localhost:8888` to see:

- **🌳 Evolutionary Tree View**: Displays the lineage relationship of ML solutions.
- **📈 Score History**: Shows the trend of model performance over iterations.
- **🔍 Code Diff**: Compares modifications between different versions of ML code.
- **🗺️ Island Map**: Visualizes multi-island evolution to maintain diversity.
- **⚡ Real-time Updates**: Automatically refreshes to display the latest evolutionary state.

## Example Projects

The project provides rich machine learning examples:

### Iris Classification Example (`ml_example`)

A complete entry-level classification task example:
- Iris dataset multi-classification.
- Standardized evaluation flow.
- Complete configuration file template.

### MLE-Bench Competition Examples

Includes multiple real-world Kaggle-style competitions:

| Category | Example Competition | Task Type | Difficulty |
|------|----------|----------|------|
| Image Classification | `aerial-cactus-identification` | Cactus Identification | Easy |
| Image Classification | `dogs-vs-cats-redux-kernels-edition` | Cats vs Dogs | Easy |
| Image Classification | `histopathologic-cancer-detection` | Cancer Detection | Easy |
| NLP | `detecting-insults-in-social-commentary` | Insult Detection | Easy |
| Tabular Data | `nomad2018-predict-transparent-conductors` | Material Prediction | Easy |
| NLP | `google-quest-challenge` | Q&A Quality Scoring | Medium |
| Tabular Data | `us-patent-phrase-to-phrase-matching` | Patent Phrase Matching | Medium |
| Time Series | `predict-volcanic-eruptions-ingv-oe` | Volcanic Eruption Prediction | Hard |
| Bioinformatics | `stanford-covid-vaccine` | mRNA Vaccine Efficacy | Hard |

## Troubleshooting

### Common Issues

1.  **Dependency Installation Issues**
    
    ```bash
    # Ensure correct Python version
    python --version  # Should be 3.12+
    
    # Reinstall dependencies
    uv sync
    ```

2.  **LLM API Configuration Errors**
    - Check URL and API Key in `llm_config`.
    - Confirm model name format is correct (e.g., `openai/gemini-3-flash-preview`).
    - Verify API endpoint accessibility.

3.  **Data File Path Errors**
    - Ensure dataset file paths are correct.
    - Check if `description.md` exists.
    - Verify data file format (CSV format).

4.  **Evaluation Timeout**
    - Check if `evaluator.timeout` setting is reasonable.
    - Optimize the performance of evaluation code.
    - Consider using GPU acceleration for the training process.

### Debugging Tips

- Use `--log-level DEBUG` to get detailed logs.
- Check evaluation records in the `output/evaluator/` directory.
- View the `agent.log` file in the task directory to understand the agent's thought process.
- Use the visualization interface to analyze evolutionary trends and bottlenecks.

## Best Practices

### Task Design

1.  **Data Preparation**
    - Ensure standardized data format (CSV format, unified encoding).
    - Provide clear data dictionaries and feature descriptions.
    - Include complete data preprocessing guidance.

2.  **Evaluation Logic**
    - Design evaluation functions to be stable and reliable.
    - Include complete error handling and boundary conditions.
    - Provide detailed evaluation metrics for analysis.

3.  **Task Description**
    - Problem definition should be clear and explicit.
    - Expected output format should be standardized.
    - Include relevant domain knowledge and constraints.

### Performance Optimization

1.  **Parameter Tuning**
    - Set iteration counts reasonably based on task complexity.
    - Adjust concurrency to balance computing resource usage.
    - Set timeout durations reasonably to avoid resource waste.

2.  **Resource Management**
    - Monitor memory and GPU usage.
    - Consider batch processing for large-scale data.
    - Optimize data loading and processing pipelines.

### Monitoring and Analysis

1.  **Progress Monitoring**
    - Periodically check log files to understand the agent's thinking process.
    - Use the visualization interface to track model performance improvement trends.
    - Analyze score changes to guide parameter adjustment strategies.

2.  **Result Analysis**
    - Save important checkpoints for subsequent in-depth analysis.
    - Compare model architecture differences and performance gains across iterations.
    - Summarize successful strategies and failure lessons for future improvements.

By following these guidelines, you can fully leverage the powerful capabilities of the ML Evolve Agent to solve complex machine learning problems, achieving full automation from data exploration to model optimization, and delivering outstanding performance in machine learning competitions like Kaggle.