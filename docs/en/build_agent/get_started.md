# Quick Start

This guide will help you quickly set up the LoongFlow development environment and run your first example.

## Requirements

### Python Version
LoongFlow requires Python 3.12 or higher.
```bash
python --version  # Confirm Python version is 3.12+
```

### Package Management
It is highly recommended to use `uv` for dependency management, as it is faster and more reliable than traditional pip/conda:
```bash
uv --version  # Confirm uv is installed
```

## Installation Steps

### 1. Clone the Repository
```bash
git clone https://github.com/baidu-baige/LoongFlow.git
cd LoongFlow
```

### 2. Create a Virtual Environment
```bash
uv venv .venv --python 3.12
source .venv/bin/activate  # Linux/macOS
# Or on Windows: .venv\Scripts\activate
```

### 3. Install Dependencies
```bash
uv pip install -e .
```

## Verify Installation

### Run Basic Tests
```bash
uv run pytest tests/ -v
```

### Simple Import Test
```python
# Create a simple test script test_import.py
from loongflow.framework.react import ReActAgent
from loongflow.agentsdk.tools import Toolkit
print("LoongFlow imported successfully!")
```

Run the test:
```bash
uv run python test_import.py
```

## Configure LLM

LoongFlow supports various models such as OpenAI, Gemini, and DeepSeek. Create a configuration file:

Create `task_config.yaml` in the project root directory:
```yaml
llm_config:
  url: "https://your-llm-api/v1"  # Your API endpoint
  api_key: "your-api-key"         # Your API key
  model: "deepseek-r1-250528"     # Recommended models: gemini-3-pro-preview or deepseek-r1-250528
```

## Run First Example

### General Evolve Agent Example
```bash
# Install example dependencies
uv pip install -r ./agents/general_evolve/examples/packing_circle_in_unit_square/requirements.txt

# Run example (background mode)
./run_task.sh packing_circle_in_unit_square --background

# View run logs
tail -f ./agents/general_evolve/examples/packing_circle_in_unit_square/run.log

# Stop task
./run_task.sh stop packing_circle_in_unit_square
```

### Machine Learning Agent Example
```bash
# Initialize ML environment
./run_ml.sh init

# Run ML example
./run_ml.sh run ml_example --background

# View run logs
tail -f ./agents/ml_evolve/examples/ml_example/agent.log

# Stop task
./run_ml.sh stop ml_example
```

## Basic Usage Examples

### Using EvolveAgent
```python
from loongflow.framework.evolve import EvolveAgent

# Configure Evolve Agent
agent = EvolveAgent(
    config=config,
    checkpoint_path=checkpoint_path,
)

# Register worker components
agent.register_planner_worker("planner", PlanAgent)
agent.register_executor_worker("executor", ExecuteAgent)
agent.register_summary_worker("summary", SummaryAgent)

# Run the agent
result = await agent()
```

### Using ReActAgent
```python
from loongflow.framework.react import AgentContext, ReActAgent
from loongflow.agentsdk.tools import Toolkit

# Build agent context
toolkit = Toolkit()
# Register tools...

# Create default ReAct agent
agent = ReActAgent.create_default(
    model=model, 
    sys_prompt=sys_prompt, 
    toolkit=toolkit
)

# Run the agent
result = await agent(message)
```