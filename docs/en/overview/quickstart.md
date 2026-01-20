# Quick Start

## Prerequisites

- **Python 3.12+** (Required)
- **Git** (For cloning the repository)
- **Recommended to use uv package manager**

## 1. Clone Repository

```bash
git clone https://github.com/baidu-baige/LoongFlow.git
cd LoongFlow
```

## 2. Environment Setup

### Using uv (Recommended)

```bash
# Create virtual environment
uv venv .venv --python 3.12
source .venv/bin/activate

# Install dependencies
uv pip install -e .
```

### Using conda

```bash
conda create -n loongflow python=3.12
conda activate loongflow
pip install -e .
```

## 3. Configure LLM

Edit the task configuration file to configure your LLM provider:

```yaml
# Example: agents/general_evolve/examples/packing_circle_in_unit_square/task_config.yaml
llm_config:
  url: "https://your-llm-api/v1"
  api_key: "your-api-key"
  model: "openai/gemini-3-pro-preview"  # Recommended model
```

## 4. Run Agents

### Run General Evolutionary Agent (Math and Algorithm Problems)

```bash
# Install task-specific dependencies
uv pip install -r ./agents/general_evolve/examples/packing_circle_in_unit_square/requirements.txt

# Run agent
./run_task.sh packing_circle_in_unit_square --background

# View logs
tail -f ./agents/general_evolve/examples/packing_circle_in_unit_square/run.log

# Stop agent
./run_task.sh stop packing_circle_in_unit_square
```

### Run Machine Learning Evolutionary Agent (Kaggle Competition Problems)

```bash
# Initialize environment
./run_ml.sh init

# Run agent
./run_ml.sh run ml_example --background

# View logs
tail -f ./agents/ml_evolve/examples/ml_example/agent.log

# Stop agent
./run_ml.sh stop ml_example
```

## 5. Custom Development

### Create Custom Task

1. **Create task directory and configuration file**

```bash
mkdir -p agents/general_evolve/examples/my_task
```

2. **Write task configuration** (`task_config.yaml`)

```yaml
llm_config:
  url: "https://your-llm-api/v1"
  api_key: "your-api-key"
  model: "openai/gemini-3-pro-preview"

evolve:
  max_iterations: 100
  target_score: 1.0
```

3. **Write evaluation code** (`evaluator.py`)

```python
def evaluate(code: str) -> float:
    # Return a score between 0.0-1.0
    return score
```

4. **Run custom task**

```bash
./run_task.sh my_task --background
```

## 6. Verify Installation

```bash
# Test framework import
python -c "import loongflow; print('Installation successful!')"

# Run basic tests
uv run pytest tests/ -v
```

## Get Help

- **GitHub Issues**: Report bugs and errors
- **Documentation**: View detailed usage guides
- **Examples**: Refer to `agents/general_evolve/examples/` and `agents/ml_evolve/examples/`

Now you can start using LoongFlow to build agents to solve complex problems!