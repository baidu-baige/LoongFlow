# ReAct Agent Framework

> Core agent engine of LoongFlow implementing the Reason-Act-Observe paradigm.

ReAct Agent Framework is the core intelligent agent engine in the LoongFlow project. It provides a highly modular
component system for building AI agents with complex reasoning capabilities, solving tasks through multi-turn iterative
execution.

## Architecture

The ReAct framework abstracts agent execution into four core components. Each component is defined as a **Protocol**
interface, with default implementations provided out of the box.

<p align="center">
  <img src="https://evolux-pub.bj.bcebos.com/share/react_agent_architecture.png" alt="ReAct Agent Architecture" width="80%"/>
</p>

### Execution Flow

1. **Reason** - Analyze current context and decide next action
2. **Act** - Execute tool calls based on reasoning output
3. **Observe** - Process action results for next iteration
4. **Finalize** - Determine task completion and construct final response

## Quick Start

### Basic Usage

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

# Create ReAct agent with default components
agent = ReActAgent.create_default(
    model=model,
    sys_prompt="You are a professional math problem solving assistant.",
    toolkit=Toolkit(),
    max_steps=10
)

result = await agent.run(Message.from_text("Solve the equation x^2 + 2x + 1 = 0"))
```

### Custom Components

```python
from evolux.react import ReActAgent, AgentContext
from evolux.react.components import Reasoner


class CustomReasoner(Reasoner):
    async def reason(self, context: AgentContext) -> Message:
        # Your custom reasoning logic
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

## Component Protocols & Default Implementations

### Reasoner

Analyzes current context and decides on the next action.

**Protocol:**

```python
class Reasoner(Protocol):
    async def reason(self, context: AgentContext) -> Message:
        """Reason the current context and decide on the next action."""
        ...
```

**Default Implementation:**

```python
from evolux.react.components import DefaultReasoner

reasoner = DefaultReasoner(
    model=llm_model,
    system_prompt="Your system prompt"
)
```

### Actor

Executes the actions decided by the Reasoner (e.g., tool calls).

**Protocol:**

```python
class Actor(Protocol):
    async def act(
        self, context: AgentContext, tool_calls: List[ToolCallElement]
    ) -> List[Message]:
        """Execute the actions decided by the Reasoner."""
        ...
```

**Default Implementations:**

| Class           | Description                     |
|-----------------|---------------------------------|
| `SequenceActor` | Execute tool calls sequentially |
| `ParallelActor` | Execute tool calls in parallel  |

```python
from evolux.react.components import SequenceActor, ParallelActor

actor = SequenceActor()  # or ParallelActor()
```

### Observer

Processes action results and prepares them for the next reasoning step.

**Protocol:**

```python
class Observer(Protocol):
    async def observe(
        self, context: AgentContext, tool_outputs: List[Message]
    ) -> Message | None:
        """Observe action results and prepare for next reasoning step."""
        ...
```

**Default Implementation:**

```python
from evolux.react.components import DefaultObserver

observer = DefaultObserver()
```

### Finalizer

Determines if the task is complete and constructs the final response.

**Protocol:**

```python
class Finalizer(Protocol):
    @property
    def answer_schema(self) -> FunctionTool:
        """The schema of the special 'final answer' tool."""
        ...

    async def resolve_answer(
        self,
        tool_call: ToolCallElement,
        tool_output: ToolOutputElement,
    ) -> Message | None:
        """Resolve a tool interaction into the final answer."""
        ...

    async def summarize_on_exceed(
        self, context: AgentContext, **kwargs
    ) -> Message | None:
        """Summarize when react loop exceeds max_steps."""
        ...
```

**Default Implementation:**

```python
from evolux.react.components import DefaultFinalizer

finalizer = DefaultFinalizer(
    model=llm_model,
    summarize_prompt="Your summarization prompt",
    output_schema=OutputModel  # Optional
)
```

## Configuration

### AgentContext

Manages agent runtime state and resources:

```python
from evolux.react import AgentContext

context = AgentContext(
    memory=grade_memory,
    toolkit=toolkit,
    max_steps=10
)
```

### Hook System

Customize execution flow at various stages:

```python
supported_hook_types = [
    "pre_run", "post_run",
    "pre_reason", "post_reason",
    "pre_act", "post_act",
    "pre_observe", "post_observe"
]
```

## Advanced Features

### Interrupt Handling

```python
async def custom_interrupt_handler(context: AgentContext):
    # Custom interrupt logic
    pass


agent.register_interrupt(custom_interrupt_handler)
```

### Memory Management

Integrates with agentsdk's GradeMemory:

- Conversation history persistence
- Execution state tracking
- Experience accumulation

### Tool Integration

Seamlessly integrates with agentsdk tool system:

- Dynamic tool registration
- Parameter validation
- Error handling

## File Structure

```
src/evolux/react/
├── components/
│   ├── base.py              # Protocol definitions
│   ├── default_reasoner.py
│   ├── default_actor.py
│   ├── default_observer.py
│   └── default_finalizer.py
├── context.py
├── react_agent_base.py
└── react_agent.py
```

## Role in LoongFlow

ReAct framework serves as the core execution engine in LoongFlow:

- **Planner Stage** - Task analysis and plan generation
- **Executor Stage** - Solution optimization through execution
- **Summary Stage** - Experience summarization and memory updates
