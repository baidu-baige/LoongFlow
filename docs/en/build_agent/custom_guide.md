# Guide to Building Custom Agents

This guide demonstrates how to build a custom Agent based on LoongFlow's ReAct architecture.

## Core Agent Components

A ReAct Agent consists of four core components:

### 1. Reasoner
Responsible for analyzing the current context and deciding the next action.

```python
from loongflow.framework.react.components import Reasoner
from loongflow.framework.react import AgentContext
from loongflow.agentsdk.message import Message

class CustomReasoner(Reasoner):
    async def reason(self, context: AgentContext) -> Message:
        # Custom reasoning logic
        # Analyze context, generate tool call decisions
        return await self.generate_reasoning(context)
```

### 2. Actor
Responsible for executing tool calls decided by the Reasoner.

```python
from loongflow.framework.react.components import Actor
from loongflow.agentsdk.message import ToolCallElement

class CustomActor(Actor):
    async def act(self, context: AgentContext, tool_calls: List[ToolCallElement]) -> List[Message]:
        # Execute tool calls serially or in parallel
        results = []
        for call in tool_calls:
            result = await context.toolkit.arun(call.target, call.arguments)
            results.append(result)
        return results
```

### 3. Observer
Optional component, used to process execution results and prepare for the next round of reasoning.

```python
from loongflow.framework.react.components import Observer

class CustomObserver(Observer):
    async def observe(self, context: AgentContext, tool_outputs: List[Message]) -> Message | None:
        # Analyze tool outputs, generate observation results
        if tool_outputs:
            return await self.analyze_results(tool_outputs)
        return None
```

### 4. Finalizer
Determines if the task is complete and constructs the final response.

```python
from loongflow.framework.react.components import Finalizer
from loongflow.agentsdk.tools import FunctionTool

class CustomFinalizer(Finalizer):
    @property
    def answer_schema(self) -> FunctionTool:
        # Define final answer tool
        return FunctionTool(
            name="final_answer",
            description="Generate final answer",
            # ... other parameters
        )
    
    async def resolve_answer(self, tool_call, tool_output) -> Message | None:
        # Resolve final answer
        if self.is_final_answer(tool_call, tool_output):
            return await self.format_final_answer(tool_output)
        return None
```

## Building the Complete Agent

### Method 1: Using Default Components

```python
from loongflow.framework.react import ReActAgent
from loongflow.agentsdk.models import BaseLLMModel

# Quickly create default Agent
agent = ReActAgent.create_default(
    model=your_llm_model,
    sys_prompt="Your system prompt",
    toolkit=your_toolkit,
    max_steps=10
)
```

### Method 2: Custom Component Composition

```python
from loongflow.framework.react import ReActAgent, AgentContext
from loongflow.agentsdk.memory.grade import GradeMemory
from loongflow.agentsdk.tools import Toolkit

# Create custom components
reasoner = CustomReasoner(model, "System prompt")
actor = CustomActor()
observer = CustomObserver()
finalizer = CustomFinalizer(model, "Summary prompt")

# Build context
memory = GradeMemory.create_default(model)
toolkit = Toolkit()  # Register your tools
context = AgentContext(memory, toolkit, max_steps=10)

# Assemble into Agent
custom_agent = ReActAgent(
    context=context,
    reasoner=reasoner,
    actor=actor, 
    observer=observer,
    finalizer=finalizer,
    name="CustomAgent"
)
```

## Tool Development

Add custom tools to the Agent:

```python
from loongflow.agentsdk.tools import BaseTool

class DomainTool(BaseTool):
    name = "domain_tool"
    description = "Domain specific tool"
    
    async def execute(self, context, **kwargs):
        # Implement tool logic
        result = await self.process_data(kwargs.get('data'))
        return ToolResponse(success=True, data=result)

# Register to toolkit
toolkit.register_tool(DomainTool())
```

## Running the Agent

```python
from loongflow.agentsdk.message import Message

# Prepare input message
initial_message = Message.from_text(
    sender="user",
    role=Role.USER, 
    data="Your task description"
)

# Run Agent
result = await agent.run(initial_message)
print(result.data)  # Get final result
```

## Configuration Best Practices

1. **Memory Management**: Use `GradeMemory` to manage conversation history
2. **Tool Registration**: Manage all tools centrally in `Toolkit`
3. **Step Control**: Set `max_steps` reasonably to prevent infinite loops
4. **Error Handling**: Implement appropriate error handling logic in each component

## Example Directory Structure

```
custom_agent/
├── __init__.py
├── agent.py              # Main Agent class
├── components/           # Custom components
│   ├── reasoner.py
│   ├── actor.py
│   ├── observer.py
│   └── finalizer.py
├── tools/               # Domain tools
│   └── domain_tools.py
└── config/             # Configuration files
    └── agent_config.yaml
```

This architecture provides a flexible component replacement mechanism, allowing you to customize the behavior of each step according to specific needs.