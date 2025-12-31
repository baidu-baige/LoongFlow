# ReAct Agent Framework

ReAct Agent
Framework 是 LoongFlow 项目中的核心智能体引擎，实现了经典的 Reason-Act-Observe（推理-执行-观察）架构。该框架提供了一套高度模块化的组件系统，支持构建具有复杂推理能力的 AI 智能体，能够通过多轮迭代的方式解决复杂的任务。

## 核心架构

ReAct 框架将智能体的执行流程抽象为四个核心组件，通过协议接口实现高度解耦：

<p align="center">
<img src="https://evolux-pub.bj.bcebos.com/share/react_agent_architecture.png" alt="ReAct Agent Architecture" width="80%"/>
</p>

### 执行流程

1. **Reason（推理）**：分析当前上下文和历史记忆，决定下一步行动策略
2. **Act（执行）**：根据推理结果调用工具或执行操作
3. **Observe（观察）**：处理执行结果，准备下一轮推理数据
4. **Finalize（终结）**：判断任务是否完成，构造最终响应

## 快速开始

### 基础使用示例

```python
from agentsdk.message import Message
from agentsdk.models import LiteLLMModel
from agentsdk.tools import Toolkit
from evolux.react import ReActAgent

# 创建LLM模型
model = LiteLLMModel(
    model_name="deepseek-r1",
    base_url="http://your-llm-service/v1",
    api_key="******"
)

toolkit = Toolkit()

# 创建默认配置的ReAct智能体
agent = ReActAgent.create_default(
    model=model,
    sys_prompt="你是一个专业的数学问题求解助手",
    toolkit=toolkit,  # 可选：工具集
    max_steps=10  # 最大迭代次数
)

# 执行任务
initial_message = Message.from_text("求解方程 x^2 + 2x + 1 = 0")
result = await agent.run(initial_message)
```

### 创建自定义 ReAct 智能体

```python
from agentsdk.message import Message
from evolux.react import ReActAgent, AgentContext
from evolux.react.components import Reasoner, Actor, Observer, Finalizer


# 自定义组件
class CustomReasoner(Reasoner):
    async def reason(self, context: AgentContext) -> Message:
        # 自定义推理逻辑
        pass


# 构建完整智能体
agent = ReActAgent(
    context=agent_context,
    reasoner=custom_reasoner,
    actor=sequence_actor,
    observer=default_observer,
    finalizer=default_finalizer,
    name="CustomAgent"
)
```

## 核心组件

### Reasoner（推理器）

**职责**：分析当前状态，规划下一步行动

```python
from evolux.react.components import DefaultReasoner

reasoner = DefaultReasoner(
    model=llm_model,
    system_prompt="系统提示词"
)
```

### Actor（执行器）

**职责**：执行工具调用，默认提供了顺序执行和并行执行功能

- `SequenceActor`：顺序执行工具调用
- `ParallelActor`：并行执行工具调用

```python
from evolux.react.components import SequenceActor, ParallelActor

# 顺序执行器
actor = SequenceActor()

# 并行执行器
actor = ParallelActor()
```

### Observer（观察器）

**职责**：处理执行结果，为下一轮推理做准备

```python
from evolux.react.components import DefaultObserver

observer = DefaultObserver()
```

### Finalizer（终结器）

**职责**：判断任务完成状态，生成最终响应

```python
from evolux.react.components import DefaultFinalizer

finalizer = DefaultFinalizer(
    model=llm_model,
    summarize_prompt="总结提示词",
    output_schema=OutputModel
)
```

## ⚙️ 配置与定制

### AgentContext（上下文管理）

管理智能体的运行状态和资源：

- **Memory**：对话历史记忆管理
- **Toolkit**：工具集管理
- **执行状态**：当前步骤、最大步骤限制

```python
from evolux.react import AgentContext

context = AgentContext(
    memory=grade_memory,
    toolkit=toolkit,
    max_steps=10
)
```

### 钩子系统

支持多种钩子类型，实现执行流程的深度定制：

```python
# 支持的钩子类型
supported_hook_types = [
    "pre_run", "post_run",
    "pre_reason", "post_reason",
    "pre_act", "post_act",
    "pre_observe", "post_observe"
]
```

## 高级特性

### 中断处理

支持智能体执行过程中的中断控制：

```python
async def custom_interrupt_handler(context: AgentContext):
    # 自定义中断逻辑
    pass


agent.register_interrupt(custom_interrupt_handler)
```

### 记忆管理

与 agentsdk 的 GradeMemory 集成，支持智能记忆管理：

- 对话历史持久化
- 执行状态跟踪
- 经验积累和学习

### 工具集成

无缝集成 agentsdk 工具系统，支持：

- 动态工具注册
- 参数验证
- 错误处理
- 批量执行

## 📁 文件结构

```
src/evolux/react/
├── components/           # 核心组件实现
│   ├── base.py          # 组件协议定义
│   ├── default_reasoner.py
│   ├── default_actor.py
│   ├── default_observer.py
│   └── default_finalizer.py
├── context.py           # 上下文管理
├── react_agent_base.py  # 智能体基类
├── react_agent.py       # 主要智能体实现
```

## 🎯 在 LoongFlow 框架中的角色

ReAct 框架是 LoongFlow 进化算法的核心执行引擎：

- **Planner 阶段**：使用 ReAct 进行任务分析和规划生成
- **Executor 阶段**：通过 ReAct 执行具体的解决方案优化
- **Summary 阶段**：运用 ReAct 进行经验总结和记忆更新

---

ReAct Agent Framework 为构建复杂的 AI 智能体提供了坚实的架构基础，通过模块化设计和协议接口，确保了框架的灵活性和可扩展性。
