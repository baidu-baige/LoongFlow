# ✨ EvolveAgent

EvolveAgent is an evolutionary paradigm designed for general algorithmic tasks. Borrowing from the exploratory work mode of human researchers, it abstracts single-round evolution into three stages: "Planner-Executor-Summary". Through evolutionary memory, it guides the model to conduct multi-round iterative exploration, continuously accumulating experience and driving improvement. This effectively enhances the quality of new solutions and the certainty of the evolutionary process, alleviating issues of ineffective evaluation and random trial-and-error, thereby automatically realizing high-efficiency, low-cost continuous evolution for complex tasks.

<p align="center"> <img src="https://evolux-pub.bj.bcebos.com/share/evolve_agent_fr_v1.png" alt="LoongFlow Evolve Framework" width="80%"/> </p>

- Planner: Responsible for fully understanding the task and global evolutionary state. Combining sampling and relevant historical evolutionary experience, it generates improvement guidance for the current iteration, providing expert-level direction.
- Executor: Responsible for implementing new solutions, performing solution evaluation, debugging errors, and conducting targeted sufficient linear optimization to produce the optimal solution under the planner's guidance.
- Summary: Responsible for a comprehensive analysis of the newly generated solutions, summarizing successful and failed experiences to guide the next evolution, and publishing the current round's evolution information to the evolutionary memory.

## 🚀 Quick Start

EvolveAgent includes a built-in implementation example, GeneralAgent, for general algorithmic task evolution. We have provided the packing_circle_in_unit_square example, which you can run directly:

```bash
# Run your first evolve task, the evolution results are in the ./output directory
./run_task.sh packing_circle_in_unit_square --background

# Stop task
./run_task.sh stop packing_circle_in_unit_square
```

### 🛠️ Self-defined Task

You can create custom evolutionary tasks by creating a new folder in the `agents/general_evolve/examples` directory. An evolutionary task must contain three files:

- `task_config.yaml` (Task Configuration): Defines task goals, LLM configuration, three-stage settings, evaluator settings, etc.
- `initial_program.py` (Initial Program): Defines the initial solution for the task, providing initial input for subsequent new solution generation. This includes necessary evaluation entry methods, fixed non-evolvable self-test methods, etc.
- `eval_program.py` (Evaluation Program): Defines the evaluator used to assess whether the new solution meets the task goals and performs scoring. The system judges whether the evolutionary task is complete based on the score.

#### Examples of task configuration

**Simple task config**:
Pick a random task_config.yaml from the examples; you only need to modify the LLM configuration and task description.

```yaml
# Global LLM configuration (Optional).
# If evaluator or other components don't have their own llm_config, this configuration will be used.
llm_config:
  url: "http://xxx/v1"
  api_key: "xxx"
  model: "deepseek-r1-250528"
  temperature: 0.8
  context_length: 128000
  max_tokens: 32768
  top_p: 1.0
  timeout: 1200
# ------------------------------------------------------------------------------
# Define the configuration for the main evolution process
# ------------------------------------------------------------------------------
evolve:
  # Task description, the core goal of the entire evolution process
  task: |
    Problem Statement: xxx
```

**Complex task config**:
You can choose a more suitable Executor, evaluation timeout, maximum evolution iterations, etc., based on the task situation.

```yaml
# Component names selected for this run
planner_name: "evolve_planner"
executor_name: "evolve_executor_fuse"
summary_name: "evolve_summary"

# Core parameters of the evolutionary process
max_iterations: 1000
target_score: 1.0
concurrency: 1

# Evaluator configuration
evaluator:
  timeout: 3600
```

#### Examples of initial_program

You must prepare a starting program for your evolutionary task. It must contain a test entry function callable by the evaluator, as well as input/output structures. Its implementation can even be an empty function. EvolveAgent will automatically fill this test entry function based on your task description—that is the charm of evolution 👏

```python
import numpy as np


def search_coefficients():
    """Find the coefficients of the problem."""
    best_coefficients = np.array([1, 2, 3])
    return best_coefficients
```

#### Examples of eval_program

The evaluator is the core of the entire evolutionary task; it determines if the new solution meets the task goals and handles scoring. You only need to modify the evaluate function and run_external_function.
**Good evaluation feedback allows the LLM to generate higher-quality solutions and accelerates evolutionary efficiency.**

For details, please refer to：[minimum_overlap_problem](../../../agents/general_evolve/examples/minimum_overlap_problem/eval_program.py)

### 📂 Directory Structure

```
├── agents
│   ├── general_evolve
│   │   ├── examples
│   │   │   ├── packing_circle_in_unit_square
│   │   │   │   ├── eval_program.py
│   │   │   │   ├── initial_program.py
│   │   │   │   └── task_config.yaml
│   │   │   ├── ...
│   │   │   └── uncertainty_inequality
│   │   │       ├── eval_program.py
│   │   │       ├── initial_program.py
│   │   │
```

## 🎩 Advance Usage

You can customize the Planner, Executor, and Summary components according to task requirements, and inject them into EvolveAgent via the register method to create your custom "EvolveAgent".

```python
from evolux.evolve import EvolveAgent

# Config evolve agent
agent = EvolveAgent(
    config=config,
    checkpoint_path=checkpoint_path,
)

# Register worker (Implement the Planner, Executor, and Summary interfaces)
agent.register_planner_worker("planner", PlanAgent)
agent.register_executor_worker("executor", ExecuteAgent)
agent.register_summary_worker("summary", SummaryAgent)

# Run agent
result = await agent()
```

### 🔧 Custom Components

The Planner, Executor, and Summary components all inherit from Worker. You only need to implement the run method. The implementation of run can be a deterministic function or a sub-Agent. For details, please refer to: [evolve_planner](../../../agents/general_evolve/evolve_planner/plan_agent.py)
