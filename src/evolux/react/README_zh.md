# ReAct Agent 框架

> LoongFlow 的核心智能体引擎，实现了 Reason-Act-Observe（推理-行动-观察）范式。

ReAct Agent 框架是 LoongFlow 项目中的核心智能体引擎。它提供了一套高度模块化的组件系统，用于构建具备复杂推理能力的 AI 智能体，通过多轮迭代执行来解决任务。

## 架构设计

ReAct 框架将智能体执行抽象为四个核心组件。每个组件都定义为 **Protocol** 接口，并提供开箱即用的默认实现。

<p align="center">
  <img src="https://evolux-pub.bj.bcebos.com/share/react_agent_architecture.png" alt="ReAct Agent 架构图" width="80%"/>
</p>

### 执行流程

1. **Reason（推理）** - 分析当前上下文，决定下一步行动
2. **Act（行动）** - 根据推理输出执行工具调用
3. **Observe（观察）** - 处理行动结果，为下一轮迭代做准备
4. **Finalize（终结）** - 判断任务是否完成，构建最终响应

## 快速开始

### 基础用法

```python
from agentsdk.message import Message
from agentsdk.models import LiteLLMModel
from agentsdk.tools import Toolkit
from evolux.react import ReActAgent

model = LiteLLMModel(
    model_name="deepseek-r1",
    base_url="http://your-llm-service/v1",
    api_key="******"
)

# 使用默认组件创建 ReAct 智能体
agent = ReActAgent.create_default(
    model=model,
    sys_prompt="你是一个专业的数学问题求解助手。",
    toolkit=Toolkit(),
    max_steps=10
)

result = await agent.run(Message.from_text("求解方程 x^2 + 2x + 1 = 0"))
```

### 自定义组件

```python
from evolux.react import ReActAgent, AgentContext
from evolux.react.components import Reasoner


class CustomReasoner(Reasoner):
    async def reason(self, context: AgentContext) -> Message:
        # 你的自定义推理逻辑
        pass


agent = ReActAgent(
    context=agent_context,
    reasoner=custom_reasoner,
    actor=sequence_actor,
    observer=default_observer,
    finalizer=default_finalizer,
    name="CustomAgent"
)
```

## 组件协议与默认实现

### Reasoner（推理器）

分析当前上下文，决定下一步行动。

**协议定义：**

```python
class Reasoner(Protocol):
    async def reason(self, context: AgentContext) -> Message:
        """推理当前上下文，决定下一步行动。"""
        ...
```

**默认实现：**

```python
from evolux.react.components import DefaultReasoner

reasoner = DefaultReasoner(
    model=llm_model,
    system_prompt="你的系统提示词"
)
```

### Actor（执行器）

执行推理器决定的行动（如工具调用）。

**协议定义：**

```python
class Actor(Protocol):
    async def act(
        self, context: AgentContext, tool_calls: List[ToolCallElement]
    ) -> List[Message]:
        """执行推理器决定的行动。"""
        ...
```

**默认实现：**

| 类名              | 描述           |
|-----------------|--------------|
| `SequenceActor` | 顺序执行工具调用     |
| `ParallelActor` | 并行执行工具调用     |

```python
from evolux.react.components import SequenceActor, ParallelActor

actor = SequenceActor()  # 或 ParallelActor()
```

### Observer（观察器）

处理行动结果，为下一轮推理做准备。

**协议定义：**

```python
class Observer(Protocol):
    async def observe(
        self, context: AgentContext, tool_outputs: List[Message]
    ) -> Message | None:
        """观察行动结果，为下一轮推理做准备。"""
        ...
```

**默认实现：**

```python
from evolux.react.components import DefaultObserver

observer = DefaultObserver()
```

### Finalizer（终结器）

判断任务是否完成，构建最终响应。

**协议定义：**

```python
class Finalizer(Protocol):
    @property
    def answer_schema(self) -> FunctionTool:
        """特殊的"最终答案"工具的 schema 定义。"""
        ...

    async def resolve_answer(
        self,
        tool_call: ToolCallElement,
        tool_output: ToolOutputElement,
    ) -> Message | None:
        """将工具交互解析为最终答案。"""
        ...

    async def summarize_on_exceed(
        self, context: AgentContext, **kwargs
    ) -> Message | None:
        """当 react 循环超过 max_steps 时进行总结。"""
        ...
```

**默认实现：**

```python
from evolux.react.components import DefaultFinalizer

finalizer = DefaultFinalizer(
    model=llm_model,
    summarize_prompt="你的总结提示词",
    output_schema=OutputModel  # 可选
)
```

## 配置说明

### AgentContext（智能体上下文）

管理智能体运行时状态和资源：

```python
from evolux.react import AgentContext

context = AgentContext(
    memory=grade_memory,
    toolkit=toolkit,
    max_steps=10
)
```

### Hook 系统

在各个执行阶段自定义执行流程：

```python
supported_hook_types = [
    "pre_run", "post_run",
    "pre_reason", "post_reason",
    "pre_act", "post_act",
    "pre_observe", "post_observe"
]
```

## 高级特性

### 中断处理

```python
async def custom_interrupt_handler(context: AgentContext):
    # 自定义中断逻辑
    pass


agent.register_interrupt(custom_interrupt_handler)
```

### 记忆管理

与 agentsdk 的 GradeMemory 集成：

- 对话历史持久化
- 执行状态追踪
- 经验积累

### 工具集成

与 agentsdk 工具系统无缝集成：

- 动态工具注册
- 参数校验
- 错误处理

## 文件结构

```
src/evolux/react/
├── components/
│   ├── base.py              # 协议定义
│   ├── default_reasoner.py
│   ├── default_actor.py
│   ├── default_observer.py
│   └── default_finalizer.py
├── context.py
├── react_agent_base.py
└── react_agent.py
```

## 在 LoongFlow 中的角色

ReAct 框架作为 LoongFlow 的核心执行引擎：

- **规划阶段** - 任务分析与计划生成
- **执行阶段** - 通过执行优化解决方案
- **总结阶段** - 经验总结与记忆更新