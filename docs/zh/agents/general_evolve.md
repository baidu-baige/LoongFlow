# General Evolve Agent - 通用进化智能体

General Evolve Agent 是 LoongFlow 框架中的核心组件，专门用于解决复杂数学问题、算法优化和开放领域的问题求解。它采用 PES (Plan-Execute-Summary) 思维范式，通过结构化思考和持续学习来驱动智能体的进化。

## 概述

General Evolve Agent 将进化算法与推理能力相结合，实现了以下几个关键特性：

- **PES 思维范式**：规划-执行-总结的循环思考过程
- **多岛进化架构**：保持解决方案的多样性，避免局部最优
- **结构化记忆系统**：积累经验知识，支持长期学习
- **实时可视化监控**：提供进化过程的完整可视化界面

## 环境准备

确保已安装 Python 3.12+ 并使用 `uv` 进行依赖管理：

```bash
# 在项目根目录执行
uv sync
```

## 任务配置

### 配置文件结构

每个任务需要创建一个 YAML 配置文件，示例结构如下：

```yaml
# 全局目录配置
workspace_path: "./output"

# LLM 配置（支持 OpenAI、Gemini、DeepSeek 等）
llm_config:
  url: "https://your-llm-api/v1"
  api_key: "your-api-key"
  model: "openai/gemini-3-pro-preview"
  temperature: 0.8
  context_length: 128000
  max_tokens: 32768

# 组件配置（规划器、执行器、总结器）
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

# 进化过程配置
evolve:
  task: "你的任务描述..."
  planner_name: "evolve_planner"
  executor_name: "evolve_executor_fuse"
  summary_name: "evolve_summary"
  max_iterations: 200
  target_score: 1.0
  concurrency: 3
  
  # 评估器配置
  evaluator:
    timeout: 1200
    
  # 数据库配置
  database:
    storage_type: "in_memory"
    num_islands: 3
    population_size: 90
    checkpoint_interval: 1
```

### 代码文件编写

建议将任务相关的代码分为三个文件：

#### 1. 初始代码 (`initial_program.py`)

包含问题的基本实现框架，作为进化过程的起点：

```python
# EVOLVE-BLOCK-START
"""你的初始算法实现"""
import numpy as np

def your_initial_solution(problem_parameters):
    # 基础实现，进化过程将基于此改进
    return solution

# EVOLVE-BLOCK-END
```

#### 2. 评估代码 (`eval_program.py`)

包含评估逻辑，用于评判进化过程中的各个解决方案：

```python
def evaluate(solution_code_path):
    """
    评估函数，返回包含 score 和状态信息的字典
    """
    try:
        # 执行解决方案并评估
        result = run_solution(solution_code_path)
        return {
            "status": "success",
            "score": calculated_score,
            "metrics": {"performance": value},
            "artifacts": {"reasoning": "详细评估结果"}
        }
    except Exception as e:
        return {
            "status": "execution_failed",
            "score": 0.0,
            "summary": f"执行失败: {str(e)}"
        }
```

#### 3. 任务描述文件

用文字详细描述问题目标和约束条件。也可以将此部分内容写着配置文件中的 `evolve` 下的 `task` 字段（可以参考 `agents/general_evolve/examples/packing_circle_in_unit_square/task_config.yaml`）。

## 运行流程

### 启动任务

使用项目提供的脚本运行任务：

```bash
# 安装任务特定依赖
uv pip install -r ./agents/general_evolve/examples/你的任务名/requirements.txt

# 启动任务（后台运行）
./run_task.sh packing_circle_in_unit_square --background

# 查看实时日志
tail -f ./agents/general_evolve/examples/packing_circle_in_unit_square/run.log

# 停止任务
./run_task.sh stop packing_circle_in_unit_square
```

### 手动运行（调试用）

如果需要更精细的控制，可以直接使用 Python 脚本：

```bash
python agents/general_evolve/general_evolve_agent.py \
  --config agents/general_evolve/examples/你的任务名/task_config.yaml \
  --initial-file agents/general_evolve/examples/你的任务名/initial_program.py \
  --eval-file agents/general_evolve/examples/你的任务名/eval_program.py \
  --max-iterations 500 \
  --log-level INFO
```

### 从检查点恢复

如果任务中断，可以从最近的检查点恢复：

```bash
python agents/general_evolve/general_evolve_agent.py \
  --config config.yaml \
  --checkpoint-path ./output/database/checkpoints/checkpoint-checkpoint-iter-89-66
```

## 输出目录结构

执行完成后，`output` 目录将包含以下结构：

```
output/
├── database/
│   └── checkpoints/
│       └── checkpoint-checkpoint-iter-{迭代数}-{编号}/
│           ├── solutions/           # 所有解决方案的JSON文件
│           ├── best_solution.json   # 最佳解决方案
│           └── metadata.json        # 元数据（最佳分数、迭代信息等）
├── 迭代编号/
│   ├── planner/                     # 规划阶段输出
│   │   ├── best_plan.txt           # 最佳规划
│   │   └── plan_{编号}.txt         # 详细规划
│   ├── executor/                    # 执行阶段输出
│   │   ├── best_solution.py        # 最佳解决方案代码
│   │   └── solution_{编号}.py      # 生成的解决方案
│   └── summarizer/                  # 总结阶段输出
│       └── best_summary.txt        # 阶段总结
└── evaluator/
    └── eval_{UUID}/                 # 评估过程记录
        ├── evaluation_result.json   # 评估结果
        └── llm_code_{UUID}.py      # 被评估的代码
```

### 输出文件说明

- **checkpoint 文件**：保存进化状态，支持断点续跑
- **solution 文件**：包含生成的代码、分数、父代信息等
- **evaluation 文件**：详细的评估过程和结果
- **日志文件**：完整的执行日志，便于调试

## 可视化监控

LoongFlow 提供实时可视化界面来监控进化过程：

### 启动可视化服务器

```bash
# 在项目根目录执行
python agents/general_evolve/visualizer/visualizer.py \
  --port 8888 \
  --checkpoint-path output/database/checkpoints
```

### 可视化功能

访问 `http://localhost:8888` 可以看到以下功能：

- **🌳 进化树视图**：显示解决方案的父子关系
- **📈 分数历史**：展示分数随迭代的变化趋势
- **🔍 代码差异**：对比不同版本的代码修改
- **🗺️ 岛屿地图**：可视化多岛进化策略
- **⚡ 实时更新**：自动刷新显示最新进化状态

### 可视化界面特性

1. **解决方案树**：以树状结构展示所有解决方案及其关系
2. **分数趋势图**：显示每代最佳分数和平均分数
3. **代码差异查看**：高亮显示代码修改内容
4. **过滤和搜索**：按分数、迭代、岛屿等条件筛选

## 示例项目

项目提供了多个示例，可以参考：

- `packing_circle_in_unit_square` - 圆形装箱问题
- `max_to_min_ratios` - 极值比率优化
- `uncertainty_inequality` - 数学不等式证明

每个示例都包含完整的配置文件和代码，可以作为新任务的参考模板。

## 故障排查

### 常见问题

1. **模块导入错误**

    ```bash
    # 确保PYTHONPATH包含项目根目录
    export PYTHONPATH=$PYTHONPATH:.
    ```

2. **LLM API 配置错误**
    - 检查 `llm_config` 中的 URL 和 API Key
    - 确认模型名称格式正确（如 `openai/gemini-3-pro-preview`）

3. **评估超时**
    - 检查 `evaluator.timeout` 设置
    - 优化评估代码的性能

### 调试技巧

- 使用 `--log-level DEBUG` 获取详细日志
- 检查 `output/evaluator/` 目录中的评估记录
- 查看可视化界面了解进化状态

## 最佳实践

1. **任务设计**
    - 明确的目标函数和约束条件
    - 合理的初始解决方案
    - 稳定的评估逻辑

2. **参数调优**
    - 根据问题复杂度设置迭代次数
    - 调整岛屿数量以平衡探索和利用
    - 合理设置超时时间

3. **监控优化**
    - 定期查看可视化界面
    - 分析分数趋势图指导参数调整
    - 保存重要检查点用于后续分析


通过遵循这些指南，你可以充分利用 General Evolve Agent 的强大能力来解决复杂的优化和算法设计问题。