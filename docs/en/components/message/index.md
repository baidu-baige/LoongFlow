# Message Component

The Message Component is the core module in the LoongFlow framework for inter-agent communication, providing a standardized message format and content element system.

## Core Concepts

### Message Roles (Role)
- **system**: System message, used to provide instructions or context.
- **user**: User message, representing user input.
- **assistant**: Assistant message, representing agent response.
- **tool**: Tool message, representing tool call results.

### MIME Type Support
- **text/plain**: Plain text content.
- **application/json**: JSON data.
- **image/jpeg**, **image/png**: Image content.
- **audio/mpeg**: Audio content.
- **video/mp4**: Video content.

## Core Classes

### Message Class
The base message class, containing complete message information:

```python
from loongflow.agentsdk.message import Message, Role, ContentElement, MimeType

# Create a basic text message
message = Message(
    role=Role.USER,
    content=[ContentElement(mime_type=MimeType.TEXT_PLAIN, data="Hello, World!")],
    sender="user123",
    trace_id="trace-001",
    conversation_id="conv-456"
)
```

### Content Elements (Elements)

#### ContentElement
General content element, supporting various media types:

```python
from loongflow.agentsdk.message import ContentElement, MimeType

# Text content
text_element = ContentElement(
    mime_type=MimeType.TEXT_PLAIN, 
    data="This is a text content"
)

# JSON content
json_element = ContentElement(
    mime_type=MimeType.APPLICATION_JSON,
    data={"key": "value", "count": 42}
)
```

#### ToolCallElement
Tool call element, used to request tool execution:

```python
from loongflow.agentsdk.message import ToolCallElement
import uuid

tool_call = ToolCallElement(
    call_id=uuid.uuid4(),
    target="weather_api",
    arguments={"city": "Beijing", "unit": "celsius"}
)
```

#### ToolOutputElement
Tool execution result element:

```python
from loongflow.agentsdk.message import ToolOutputElement, ContentElement, ToolStatus
import uuid

tool_output = ToolOutputElement(
    call_id=uuid.uuid4(),  # Must match the call_id of the corresponding ToolCallElement
    tool_name="calculator",
    status=ToolStatus.SUCCESS,
    result=[ContentElement(mime_type=MimeType.TEXT_PLAIN, data="Calculation result: 42")]
)
```

#### ThinkElement
Agent thinking process element:

```python
from loongflow.agentsdk.message import ThinkElement

think_element = ThinkElement(
    content="I need to analyze the user's question and formulate a solution strategy..."
)
```

#### EvolveResultElement
Evolution process result element (used in evolutionary algorithm scenarios):

```python
from loongflow.agentsdk.message import EvolveResultElement

evolve_result = EvolveResultElement(
    best_score=0.95,
    best_solution="Optimized algorithm code",
    evaluation="Performed well on the test set",
    start_time="2024-01-01 10:00:00",
    end_time="2024-01-01 11:00:00",
    cost_time=3600,
    last_iteration=50,
    total_iterations=100
)
```

## Convenience Constructors

The Message class provides various convenience constructor methods:

```python
from loongflow.agentsdk.message import Message, Role

# Create message from text
text_message = Message.from_text("Hello, this is a text message")

# Create message from content
json_message = Message.from_content(
    data={"result": "success"}, 
    mime_type="application/json"
)

# Create message from tool call
tool_message = Message.from_tool_call(
    target="search_engine",
    arguments={"query": "Artificial Intelligence"}
)

# Create message from tool execution result
tool_output_message = Message.from_tool_output(
    call_id=call_id,           # Mandatory call_id parameter
    tool_name="search_engine", # Mandatory tool_name parameter
    status="success",
    result=[ContentElement(mime_type="text/plain", data="Search results")],
    sender="search_service"
)

# Create message from thinking content
think_message = Message.from_think("I am thinking about a solution...")

# Create message from element list
elements_message = Message.from_elements([
    ContentElement(mime_type="text/plain", data="Part one"),
    ThinkElement(content="Thinking process")
])

# Create message from media content (dedicated method)
media_message = Message.from_media(
    sender="camera_sensor",
    mime_type="image/jpeg",
    data=image_data,
    role=Role.TOOL
)
```

## Message Operations

### Serialization and Deserialization
```python
# Serialize to dictionary
message_dict = message.to_dict()

# Create message from dictionary
new_message = Message.from_dict(message_dict)
```

### Element Filtering
```python
# Get all content elements in the message
content_elements = message.get_elements(ContentElement)

# Get thinking elements
think_elements = message.get_elements(ThinkElement)

# Get evolution result elements
evolve_elements = message.get_elements(EvolveResultElement)
```

## Usage Examples

### Complete Conversation Flow
```python
from loongflow.agentsdk.message import Message, Role, ContentElement, ToolCallElement, ToolOutputElement
import uuid

# User asks a question
user_message = Message.from_text(
    "How is the weather in Beijing today?", 
    role=Role.USER,
    sender="user123"
)

# Agent requests tool call
call_id = uuid.uuid4()
agent_message = Message.from_tool_call(
    target="weather_service",
    arguments={"city": "Beijing"},
    sender="weather_agent",
    role=Role.ASSISTANT
)

# Tool execution result
tool_message = Message.from_tool_output(
    call_id=call_id,
    tool_name="weather_service",  # Must specify tool name
    status="success",
    result=[ContentElement(mime_type="text/plain", data="Beijing: Sunny, 25°C")],
    sender="weather_service",
    role=Role.TOOL
)
```

### Message with Thinking Process
```python
from loongflow.agentsdk.message import Message, ThinkElement, ContentElement

# Agent's thinking + response message
reasoning_message = Message.from_elements([
    ThinkElement(content="User wants to know weather info, I need to call weather service"),
    ContentElement(mime_type="text/plain", data="Let me check the weather in Beijing...")
], sender="assistant", role=Role.ASSISTANT)
```

### Evolution Algorithm Result Message
```python
from loongflow.agentsdk.message import Message, EvolveResultElement

# Evolutionary algorithm execution result
evolve_message = Message.from_elements([
    EvolveResultElement(
        best_score=0.98,
        best_solution="Final mathematical formula",
        evaluation="Reached 98% accuracy on the validation set",
        start_time="10:00:00",
        end_time="11:30:00",
        cost_time=5400,
        last_iteration=25,
        total_iterations=100
    )
], sender="evolve_agent", role=Role.ASSISTANT)
```

## Best Practices

1. **Use appropriate roles**: Choose the correct `Role` enum value based on the message source.
2. **Structured content**: Use appropriate element types to organize message content.
3. **Include trace information**: Use `trace_id` and `conversation_id` to trace message flows.
4. **Utilize metadata**: Store additional context information in the `metadata` field.
5. **Correct tool call matching**: Ensure the `call_id` of `ToolOutputElement` matches the corresponding `ToolCallElement`.

The Message Component provides powerful and flexible messaging capabilities for the LoongFlow framework, supporting complex agent collaboration scenarios.