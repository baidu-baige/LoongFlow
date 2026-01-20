# 快速开始

## 前提条件

- **Python 3.12+** (必须)
- **Git** (用于克隆仓库)
- **推荐使用 uv 包管理器**

## 1. 克隆仓库

```bash
git clone https://github.com/baidu-baige/LoongFlow.git
cd LoongFlow
```

## 2. 环境设置

### 使用 uv (推荐)

```bash
# 创建虚拟环境
uv venv .venv --python 3.12
source .venv/bin/activate

# 安装依赖
uv pip install -e .
```

### 使用 conda

```bash
conda create -n loongflow python=3.12
conda activate loongflow
pip install -e .
```

## 3. 配置 LLM

编辑任务配置文件，配置您的 LLM 提供商：

```yaml
# 示例：agents/general_evolve/examples/packing_circle_in_unit_square/task_config.yaml
llm_config:
  url: "https://your-llm-api/v1"
  api_key: "your-api-key"
  model: "openai/gemini-3-pro-preview"  # 推荐模型
```

## 4. 运行智能体

### 运行通用进化智能体 (数学和算法问题)

```bash
# 安装任务特定依赖
uv pip install -r ./agents/general_evolve/examples/packing_circle_in_unit_square/requirements.txt

# 运行智能体
./run_task.sh packing_circle_in_unit_square --background

# 查看日志
tail -f ./agents/general_evolve/examples/packing_circle_in_unit_square/run.log

# 停止智能体
./run_task.sh stop packing_circle_in_unit_square
```

### 运行机器学习进化智能体 (Kaggle竞赛问题)

```bash
# 初始化环境
./run_ml.sh init

# 运行智能体
./run_ml.sh run ml_example --background

# 查看日志
tail -f ./agents/ml_evolve/examples/ml_example/agent.log

# 停止智能体
./run_ml.sh stop ml_example
```

## 5. 自定义开发

### 创建自定义任务

1. **创建任务目录和配置文件**

```bash
mkdir -p agents/general_evolve/examples/my_task
```

2. **编写任务配置** (`task_config.yaml`)

```yaml
llm_config:
  url: "https://your-llm-api/v1"
  api_key: "your-api-key"
  model: "openai/gemini-3-pro-preview"

evolve:
  max_iterations: 100
  target_score: 1.0
```

3. **编写评估代码** (`evaluator.py`)

```python
def evaluate(code: str) -> float:
    # 返回 0.0-1.0 的评分
    return score
```

4. **运行自定义任务**

```bash
./run_task.sh my_task --background
```

## 6. 验证安装

```bash
# 测试框架导入
python -c "import loongflow; print('安装成功!')"

# 运行基础测试
uv run pytest tests/ -v
```

## 获取帮助

- **GitHub Issues**: 报告问题和错误
- **文档**: 查看详细的使用指南
- **示例**: 参考 `agents/general_evolve/examples/` 和 `agents/ml_evolve/examples/`

现在您可以开始使用 LoongFlow 来构建智能体解决复杂问题了！