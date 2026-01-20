# 构建自定义 Agent 指南

本指南演示如何基于 LoongFlow 的 ReAct 架构构建自定义 Agent。

## Agent 核心组件

ReAct Agent 由四个核心组件构成：

### 1. Reasoner（推理器）
负责分析当前上下文并决定下一步动作。

```python
from loongflow.framework.react.components import Reasoner
from loongflow.framework.react import AgentContext
from loongflow.agentsdk.message import Message

class CustomReasoner(Reasoner):
    async def reason(self, context: AgentContext) -> Message:
        # 自定义推理逻辑
        # 分析上下文，生成工具调用决策
        return await self.generate_reasoning(context)
```

### 2. Actor（执行器）
负责执行推理器决定的工具调用。

```python
from loongflow.framework.react.components import Actor
from loongflow.agentsdk.message import ToolCallElement

class CustomActor(Actor):
    async def act(self, context: AgentContext, tool_calls: List[ToolCallElement]) -> List[Message]:
        # 串行或并行执行工具调用
        results = []
        for call in tool_calls:
            result = await context.toolkit.arun(call.target, call.arguments)
            results.append(result)
        return results
```

### 3. Observer（观察器）
可选组件，用于处理执行结果并为下一轮推理准备。

```python
from loongflow.framework.react.components import Observer

class CustomObserver(Observer):
    async def observe(self, context: AgentContext, tool_outputs: List[Message]) -> Message | None:
        # 分析工具输出，生成观察结果
        if tool_outputs:
            return await self.analyze_results(tool_outputs)
        return None
```

### 4. Finalizer（终结器）
判断任务是否完成并构造最终响应。

```python
from loongflow.framework.react.components import Finalizer
from loongflow.agentsdk.tools import FunctionTool

class CustomFinalizer(Finalizer):
    @property
    def answer_schema(self) -> FunctionTool:
        # 定义最终答案工具
        return FunctionTool(
            name="final_answer",
            description="生成最终答案",
            # ... 其他参数
        )
    
    async def resolve_answer(self, tool_call, tool_output) -> Message | None:
        # 解析最终答案
        if self.is_final_answer(tool_call, tool_output):
            return await self.format_final_answer(tool_output)
        return None
```

## 构建完整 Agent

### 方法一：使用默认组件

```python
from loongflow.framework.react import ReActAgent
from loongflow.agentsdk.models import BaseLLMModel

# 快速创建默认 Agent
agent = ReActAgent.create_default(
    model=your_llm_model,
    sys_prompt="你的系统提示词",
    toolkit=your_toolkit,
    max_steps=10
)
```

### 方法二：自定义组件组合

```python
from loongflow.framework.react import ReActAgent, AgentContext
from loongflow.agentsdk.memory.grade import GradeMemory
from loongflow.agentsdk.tools import Toolkit

# 创建自定义组件
reasoner = CustomReasoner(model, "系统提示词")
actor = CustomActor()
observer = CustomObserver()
finalizer = CustomFinalizer(model, "总结提示词")

# 构建上下文
memory = GradeMemory.create_default(model)
toolkit = Toolkit()  # 注册你的工具
context = AgentContext(memory, toolkit, max_steps=10)

# 组合成 Agent
custom_agent = ReActAgent(
    context=context,
    reasoner=reasoner,
    actor=actor, 
    observer=observer,
    finalizer=finalizer,
    name="自定义Agent"
)
```

## 工具开发

为 Agent 添加自定义工具：

```python
from loongflow.agentsdk.tools import BaseTool

class DomainTool(BaseTool):
    name = "domain_tool"
    description = "领域特定工具"
    
    async def execute(self, context, **kwargs):
        # 实现工具逻辑
        result = await self.process_data(kwargs.get('data'))
        return ToolResponse(success=True, data=result)

# 注册到工具包
toolkit.register_tool(DomainTool())
```

## 运行 Agent

```python
from loongflow.agentsdk.message import Message

# 准备输入消息
initial_message = Message.from_text(
    sender="user",
    role=Role.USER, 
    data="你的任务描述"
)

# 运行 Agent
result = await agent.run(initial_message)
print(result.data)  # 获取最终结果
```

## 配置最佳实践

1. **内存管理**: 使用 `GradeMemory` 管理对话历史
2. **工具注册**: 在 `Toolkit` 中统一管理所有工具
3. **步数控制**: 合理设置 `max_steps` 防止无限循环
4. **错误处理**: 在各组件中实现适当的错误处理逻辑

## 示例目录结构

```
custom_agent/
├── __init__.py
├── agent.py              # 主 Agent 类
├── components/           # 自定义组件
│   ├── reasoner.py
│   ├── actor.py
│   ├── observer.py
│   └── finalizer.py
├── tools/               # 领域工具
│   └── domain_tools.py
└── config/             # 配置文件
    └── agent_config.yaml
```

这个架构提供了灵活的组件替换机制，让你可以根据具体需求定制每个环节的行为。