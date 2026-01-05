# LoongFlow

## Project Context
**LoongFlow** is an Evolutionary Agent Development Framework designed to build autonomous agents that improve through a Plan-Execute-Summary (PES) paradigm.
It moves beyond simple ReAct loops to structured evolutionary processes, focusing on high efficiency and stability for complex tasks like mathematical discovery and machine learning competitions (AutoML).

## Repository Overview

### Core Framework (`src/`)
- **`src/evolux/`**: The core logic for agent architectures.
  - `evolve/`: Implementation of `EvolveAgent` (Evolutionary algorithms).
  - `react/`: Implementation of `ReActAgent` (Standard reasoning loops).
  - `base/`: Abstract base classes.
- **`src/agentsdk/`**: Foundational building blocks.
  - `tools/`: Tool interfaces and implementations.
  - `memory/`: Memory management for agents.
  - `logger/`, `message/`, `models/`, `token/`: Utility modules.

### Agent Implementations (`agents/`)
- **`agents/general_evolve/`**: Agents for open-ended math and algorithm optimization.
  - Contains `evolve_planner`, `evolve_executor`, `evolve_summary` modules.
  - **Examples**: Located in `agents/general_evolve/examples/`.
- **`agents/ml_evolve/`**: Agents specialized for Machine Learning tasks (Kaggle/MLE-Bench).
  - Contains `evocoder`, `evaluator` and specific ML workflow logic.
  - **Examples**: Located in `agents/ml_evolve/examples/`.

### Configuration & Scripts
- **`pyproject.toml`**: Project dependencies and configuration.
- **`uv.lock`**: Dependency lock file (managed by `uv`).
- **`run_task.sh`**: Entry point for General Evolve tasks.
- **`run_ml.sh`**: Entry point for ML Evolve tasks.

## Tech Stack & Dependencies
- **Language**: Python 3.12+ (Strict requirement).
- **Package Manager**: `uv` (Preferred over pip/conda).
- **LLM Integration**: Supports OpenAI, Gemini, DeepSeek via config.
- **Core Libraries**: Pydantic (data validation), PyTorch (implied for ML tasks).

## Development Workflow

### 1. Environment Setup
Always use `uv` for dependency management.
```bash
uv venv .venv --python 3.12
source .venv/bin/activate
uv pip install -e .
```

### 2. Running Agents
Do not run python scripts directly if a shell wrapper exists.

**General Evolve Agent:**
```bash
# Install task-specific deps first
uv pip install -r ./agents/general_evolve/examples/<task_name>/requirements.txt
# Run task
./run_task.sh <task_name> --background
# Logs are found in the example directory's run.log
```

**ML Evolve Agent:**
```bash
./run_ml.sh init
./run_ml.sh run <task_name> --background
# Logs are found in the example directory's agent.log
```

### 3. Configuration
- LLM configurations are stored in `task_config.yaml` within specific example directories.
- Ensure `llm_config` follows the schema: `url`, `api_key`, `model`.

## Coding Guidelines

### Architecture Patterns
- **PES Paradigm**: When designing agents, strictly separate concerns into **Planner** (strategy), **Executor** (action/coding), and **Summary** (evaluation/learning).
- **Modularity**: New tools should be added to `agentsdk/tools` if generic, or kept local to the agent if specific.

### Style & Standards
- **Type Hinting**: Enforce Python type hints for all public interfaces.
- **Asynchronous**: The framework relies heavily on `async/await`. Ensure `run()` methods are awaitable.
- **Path Handling**: Use relative paths carefully; agent execution contexts often depend on the root directory.

### Testing
- Tests are located in `tests/` mirroring the `src/` structure.
- Run tests using:
  ```bash
  uv run pytest
  ```

## Thinking Process for AI
1. **Identify the Scope**: Are you modifying the Core Framework (`src/evolux`) or a Specific Agent (`agents/`)?
2. **Respect the Abstraction**: Do not hardcode agent logic into `agentsdk`. Keep the SDK generic.
3. **Check Dependencies**: If adding a new library, check if it belongs in `pyproject.toml` (core) or a specific example's `requirements.txt`.
4. **Safety**: When modifying `executor` modules, ensure code execution sandboxing or warnings are preserved, as these agents generate and run code.