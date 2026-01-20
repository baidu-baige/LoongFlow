# 快速开始

本指南将帮助您快速搭建 LoongFlow 开发环境并运行第一个示例。

## 环境要求

### Python 版本
LoongFlow 需要 Python 3.12 或更高版本。
```bash
python --version  # 确认 Python 版本为 3.12+
```

### 包管理
强烈推荐使用 `uv` 进行依赖管理，它比传统的 pip/conda 更快更可靠：
```bash
uv --version  # 确认 uv 已安装
```

## 安装步骤

### 1. 克隆代码库
```bash
git clone https://github.com/baidu-baige/LoongFlow.git
cd LoongFlow
```

### 2. 创建虚拟环境
```bash
uv venv .venv --python 3.12
source .venv/bin/activate  # Linux/macOS
# 或在 Windows 上: .venv\Scripts\activate
```

### 3. 安装依赖
```bash
uv pip install -e .
```

## 验证安装

### 运行基础测试
```bash
uv run pytest tests/ -v
```

### 简单导入测试
```python
# 创建一个简单的测试脚本 test_import.py
from loongflow.framework.react import ReActAgent
from loongflow.agentsdk.tools import Toolkit
print("LoongFlow 导入成功！")
```

运行测试：
```bash
uv run python test_import.py
```

## 配置 LLM

LoongFlow 支持 OpenAI、Gemini、DeepSeek 等多种模型。创建配置文件：

在项目根目录创建 `task_config.yaml`：
```yaml
llm_config:
  url: "https://your-llm-api/v1"  # 您的 API 端点
  api_key: "your-api-key"         # 您的 API 密钥
  model: "deepseek-r1-250528"     # 推荐模型：gemini-3-pro-preview 或 deepseek-r1-250528
```

## 运行第一个示例

### 通用进化智能体示例
```bash
# 安装示例依赖
uv pip install -r ./agents/general_evolve/examples/packing_circle_in_unit_square/requirements.txt

# 运行示例（后台模式）
./run_task.sh packing_circle_in_unit_square --background

# 查看运行日志
tail -f ./agents/general_evolve/examples/packing_circle_in_unit_square/run.log

# 停止任务
./run_task.sh stop packing_circle_in_unit_square
```

### 机器学习智能体示例
```bash
# 初始化 ML 环境
./run_ml.sh init

# 运行 ML 示例
./run_ml.sh run ml_example --background

# 查看运行日志
tail -f ./agents/ml_evolve/examples/ml_example/agent.log

# 停止任务
./run_ml.sh stop ml_example
```

## 基础使用示例

### 使用 EvolveAgent
```python
from loongflow.framework.evolve import EvolveAgent

# 配置进化智能体
agent = EvolveAgent(
    config=config,
    checkpoint_path=checkpoint_path,
)

# 注册工作组件
agent.register_planner_worker("planner", PlanAgent)
agent.register_executor_worker("executor", ExecuteAgent)
agent.register_summary_worker("summary", SummaryAgent)

# 运行智能体
result = await agent()
```

### 使用 ReActAgent
```python
from loongflow.framework.react import AgentContext, ReActAgent
from loongflow.agentsdk.tools import Toolkit

# 构建智能体上下文
toolkit = Toolkit()
# 注册工具...

# 创建默认的 ReAct 智能体
agent = ReActAgent.create_default(
    model=model, 
    sys_prompt=sys_prompt, 
    toolkit=toolkit
)

# 运行智能体
result = await agent(message)
```
