# Tool Components

LoongFlow Tool Components provide a secure and scalable tool invocation framework for agents, supporting common functions such as file operations, code execution, and Shell commands. It is the core infrastructure for building evolutionary agents.

## Architecture Design

### Core Class Hierarchy

```
BaseTool (Abstract Base Class)
    ├── FunctionTool (Function Tool Base Class)
    │   ├── ReadTool (File Read Tool)
    │   ├── WriteTool (File Write Tool)  
    │   ├── LsTool (Directory List Tool)
    │   ├── ShellTool (Shell Command Execution Tool)
    │   ├── ExecuteCodeTool (Code Execution Tool)
    │   ├── AgentTool (Agent Invocation Tool)
    │   ├── TodoReadTool (Todo Read Tool)
    │   ├── TodoWriteTool (Todo Write Tool)
    │   └── Custom Tools
    └── Toolkit (Toolkit Manager)
```

### Core Concepts

- **ToolContext**: Tool execution context, containing authentication status and runtime information.
- **ToolResponse**: Standardized tool execution response format.
- **FunctionDeclaration**: Tool function declaration, used for LLM function calling.

## Core Classes in Detail

### BaseTool - Abstract Base Tool

```python
from loongflow.agentsdk.tools import BaseTool
from loongflow.agentsdk.tools.tool_context import ToolContext
from loongflow.agentsdk.tools.tool_response import ToolResponse

class CustomTool(BaseTool):
    """Custom tool example"""
    
    def __init__(self, *, name, description):
        super().__init__(name=name, description=description)
    
    def get_declaration(self) -> Optional[FunctionDeclarationDict]:
        """Returns the tool's function declaration"""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "input": {"type": "string", "description": "Input parameter"}
                },
                "required": ["input"]
            }
        }
    
    async def arun(self, *, args: Dict[str, Any], tool_context: Optional[ToolContext] = None) -> ToolResponse:
        """Execute tool asynchronously"""
        # Implement tool logic
        return ToolResponse(content=[...])
```

### FunctionTool - Function Tool Implementation

Tool implementation based on Pydantic models, supporting parameter validation:

```python
from pydantic import BaseModel, Field
from loongflow.agentsdk.tools import FunctionTool

class CalculatorArgs(BaseModel):
    expression: str = Field(..., description="Mathematical expression")
    
class CalculatorTool(FunctionTool):
    def __init__(self):
        super().__init__(
            args_schema=CalculatorArgs,
            name="calculator",
            description="Execute mathematical calculations"
        )
    
    def run(self, *, args, tool_context=None):
        validated_args = self._prepare_call_args(args, tool_context)
        result = eval(validated_args["expression"])
        return ToolResponse(content=[...])
```

### Toolkit - Toolkit Manager

```python
from loongflow.agentsdk.tools import Toolkit, ReadTool, WriteTool

# Create toolkit
toolkit = Toolkit()

# Register tools (supports authentication configuration)
toolkit.register_tool(ReadTool())
toolkit.register_tool(WriteTool())

# Get tool declarations (returns OpenAI function call format)
declarations = toolkit.get_declarations()

# Synchronous tool execution
response = toolkit.run("Read", args={"file_path": "/path/to/file"})

# Asynchronous tool execution
response = await toolkit.arun("Read", args={"file_path": "/path/to/file"})
```

## Built-in Tools

### ReadTool - File Read Tool

Tool Name: **`Read`**

```python
from loongflow.agentsdk.tools import ReadTool

read_tool = ReadTool()
response = await read_tool.arun({
    "file_path": "/absolute/path/to/file.py",
    "offset": 10,      # Optional: start line number
    "limit": 20        # Optional: number of lines to read
})

# Return format: ToolResponse
# - Text file: Returns formatted text content (with line numbers)
# - Image file: Returns {"type": "image", "path": file_path}
# - PDF file: Returns {"type": "pdf", "path": file_path}
# - Notebook: Returns {"type": "notebook", "path": file_path}

# Supported file types:
# - Text files (.txt, .py, .json, .csv, .md, .html, .xml)
# - Image files (.png, .jpg, .jpeg, .gif) 
# - PDF files (.pdf)
# - Jupyter Notebook (.ipynb)
```

### WriteTool - File Write Tool

Tool Name: **`Write`**

```python
from loongflow.agentsdk.tools import WriteTool

write_tool = WriteTool()
response = await write_tool.arun({
    "file_path": "relative/or/absolute/path.txt",
    "content": "File content"
})

# Features:
# - Automatically creates directory structure
# - Supports relative paths (relative to project root, resolves to project root)
# - Overwrites existing files
# - Return format: ToolResponse containing success message
```

### TodoReadTool - Todo Read Tool

Tool Name: **`TodoRead`**

```python
from loongflow.agentsdk.tools import TodoReadTool

read_todo = TodoReadTool()
response = await read_todo.arun({})

# Return structure (in ToolResponse.content[0].data):
# {
#   "message": "Todo list retrieved successfully.",
#   "file_path": "./todo_list.json",
#   "todos": [
#     {
#       "id": "uuid",
#       "content": "Task description", 
#       "status": "pending"
#     }
#   ]
# }

# Features:
# - Automatically gets file path from ToolContext or uses default
# - Supports custom todo file storage location
# - No-parameter invocation, easy to use
```

### TodoWriteTool - Todo Write Tool

Tool Name: **`TodoWrite`**

```python
from loongflow.agentsdk.tools import TodoWriteTool

write_tool = TodoWriteTool()
response = await write_tool.arun({
    "todos": [
        {
            "content": "Complete Task A",
            "status": "completed"
        },
        {
            "content": "Start Task B", 
            "status": "in_progress"
        }
    ]
})

# Return structure (in ToolResponse.content[0].data):
# {
#   "message": "Todo list updated successfully.",
#   "file_path": "./todo_list.json",
#   "todos": [...]
# }

# Features:
# - Supports custom todo items and status management
# - Automatically creates non-existent directory structures
# - Shares file path configuration with TodoReadTool
```

### LsTool - Directory List Tool

Tool Name: **`LS`**

```python
from loongflow.agentsdk.tools import LsTool

ls_tool = LsTool()
response = await ls_tool.arun({
    "path": "/absolute/path/to/directory",
    "ignore": ["*.pyc", "__pycache__"]  # Optional: ignore patterns
})

# Return structure (in ToolResponse.content[0].data):
# {
#   "files": [
#     {
#       "name": "filename",
#       "path": "/full/path",
#       "relative_path": "relative_name",
#       "is_dir": false,
#       "size": 1024
#     }
#   ]
# }

# Features:
# - Supports automatic path conversion (relative to absolute)
# - Supports ignore pattern filtering
# - Traverses only one directory level
```

### ShellTool - Shell Command Execution Tool

Tool Name: **`ShellTool`**

```python
from loongflow.agentsdk.tools import ShellTool

shell_tool = ShellTool()
response = await shell_tool.arun({
    "commands": [
        {"command": "ls -la", "dir": "/working/directory"},
        {"command": "python script.py"}
    ]
})

# Return structure for each command:
# {
#   "command": "Executed command",
#   "dir": "Working directory",
#   "returncode": Exit code,
#   "stdout": "Standard output",
#   "stderr": "Standard error"
# }

# Features:
# - Supports asynchronous and synchronous execution
# - Batch command execution
# - Error handling and result return
```

### ExecuteCodeTool - Code Execution Tool

Tool Name: **`ExecuteCode`**

```python
from loongflow.agentsdk.tools import ExecuteCodeTool

code_tool = ExecuteCodeTool()

# Execute inline code
response = await code_tool.arun({
    "mode": "code",
    "code": "print('Hello World')",
    "language": "python",  # Optional, default is "python"
    "timeout": 30
})

# Execute file
response = await code_tool.arun({
    "mode": "file", 
    "file_path": "/path/to/script.py",
    "language": "python",
    "timeout": 60
})

# Return structure (in ToolResponse.content[0].data):
# {
#   "stdout": "Standard output",
#   "stderr": "Standard error", 
#   "returncode": Exit code,
#   "error": "Error message",
#   "execution_time": Execution time (seconds)
# }

# Features:
# - Supports Python code and script files
# - Timeout control
# - Subprocess isolation execution
```

### AgentTool - Agent Invocation Tool

Tool Name: **Inherited from the passed agent's `name` attribute**

Encapsulates an `AgentBase` agent as a callable tool, supporting collaborative calls between agents:

```python
from loongflow.agentsdk.tools import AgentTool
from loongflow.agentsdk.tools.tool_response import ToolResponse
from loongflow.agentsdk.message.message import Message, ContentElement, MimeType
from pydantic import BaseModel, Field
from loongflow.framework.base.agent_base import AgentBase

# Define agent input schema
class MyAgentInput(BaseModel):
    request: str = Field(..., description="User request text")

# Create agent
class MyAgent(AgentBase):
    name = "my_agent"
    description = "Example Agent"
    input_schema = MyAgentInput
    
    async def run(self, request: str) -> Message:
        """Agent main logic implementation"""
        # Process request and return message
        response_text = f"Processing result: {request}"
        return Message.from_text(
            data=response_text,
            sender=self.name,
            role="assistant",
            mime_type=MimeType.TEXT_PLAIN,
        )
    
    async def interrupt_impl(self) -> Message:
        """Interrupt handling logic"""
        return Message.from_text("Agent interrupted")

# Create AgentTool wrapper
agent = MyAgent()
agent_tool = AgentTool(agent)

# Call within other agents (Correct way)
response = await agent_tool.arun(args={
    "request": "Detailed description of this task"
}, tool_context=None)

# Or use synchronous call
response = agent_tool.run(args={
    "request": "Detailed description of this task"
}, tool_context=None)

# The returned ToolResponse contains ContentElement converted from Message
if response.content:
    content = response.content[0]
    print(f"Response type: {content.mime_type}")
    print(f"Response data: {content.data}")

# Features:
# - **Tool Name**: Inherits from the agent's `name` attribute
# - **Parameter Schema**: Based on the agent's `input_schema`; uses default `{"request": "string"}` if not present
# - **Message Conversion**: Message returned by the agent is converted to ToolResponse, extracting ContentElement content
# - **Call Mode**: Supports synchronous `run()` and asynchronous `arun()` calls
# - **Declaration Generation**: Automatically generates JSON Schema for LLM function calling
```

### AgentTool Function Declaration Detail

AgentTool automatically generates function declarations. Example structure:

```json
{
  "name": "echo_agent",
  "description": "Echoes the user input back as text.",
  "parameters": {
    "type": "object",
    "properties": {
      "request": {
        "type": "string", 
        "description": "User input text to be echoed."
      }
    },
    "required": ["request"]
  }
}
```

### Advanced Usage: Message Processing Mechanism

AgentTool's `_wrap_message_as_response()` method intelligently handles message conversion:

- **ContentElement Extraction**: Prioritizes extracting all ContentElements from the Message.
- **Fallback Mechanism**: If no ContentElement exists, serializes the entire Message to JSON.
- **Message Type Support**: Supports various MIME types like text, JSON, images, etc.

```python
# Custom message conversion logic example
message = await agent(**args)
response = AgentTool._wrap_message_as_response(message)
print(f"Number of content elements after conversion: {len(response.content)}")
```

## Authentication and Context Management

### ToolContext Authentication Management

```python
from loongflow.agentsdk.tools.tool_context import ToolContext, AuthConfig, AuthCredential, AuthType

# Create authentication configuration
auth_config = AuthConfig(scheme=AuthType.API_KEY, key="openai_api")

# Create authentication credential
credential = AuthCredential(auth_type=AuthType.API_KEY, api_key="sk-...")

# Set authentication in tool context
context = ToolContext(function_call_id="call_123")
context.set_auth(auth_config, credential)

# Use authentication context during tool execution
response = await tool.arun(args={}, tool_context=context)
```

### Toolkit Authentication Integration and Management Methods

```python
from loongflow.agentsdk.tools import Toolkit, AuthConfig, AuthCredential, AuthType

# Create toolkit
toolkit = Toolkit()

# 1. Add authentication when registering tool (Recommended)
toolkit.register_tool(
    api_tool, 
    auths=[(auth_config, credential)]
)

# 2. Set authentication separately (for registered tools)
toolkit.set_auth("tool_name", auth_config, credential)

# 3. Get tool authentication
stored_cred = toolkit.get_auth("tool_name", auth_config)

# 4. Get tool instance
tool = toolkit.get("tool_name")

# 5. List all registered tools
tool_names = toolkit.list_tools()

# 6. Unregister tool
toolkit.unregister_tool("tool_name")

# 7. Get tool context
context = toolkit.get_context("tool_name")

# 8. Get all tool declarations (returns list in OpenAI function call format)
# Format: [{"type": "function", "function": {...tool_declaration...}}, ...]
declarations = toolkit.get_declarations()
```

### Advanced Usage: Context Management and Execution Control

```python
from loongflow.agentsdk.tools import Toolkit, ToolContext

# Execute tool with custom context
context = ToolContext(function_call_id="custom_session", state={"custom_data": "value"})
response = await toolkit.arun("tool_name", args={"param": "value"}, tool_context=context)

# Ensure tool context exists (internal method)
tool_context = toolkit.ensure_context("tool_name", external_context=context)

# Batch tool execution
async def execute_tools_sequentially(toolkit, operations: list):
    results = []
    for tool_name, args_dict in operations:
        result = await toolkit.arun(tool_name, args=args_dict)
        results.append((tool_name, result))
        if result.err_msg:
            break  # Stop on error
    return results

# Error handling and tool chain
try:
    operations = [
        ("LS", {"path": "./"}),
        ("Read", {"file_path": "./README.md"}),
        ("ExecuteCode", {"mode": "code", "code": "print('test')"})
    ]
    results = await execute_tools_sequentially(toolkit, operations)
except Exception as e:
    print(f"Tool chain execution failed: {e}")
```

## Actual Usage Patterns

### Tool Construction Patterns in Projects

LoongFlow projects typically use the `FunctionTool(func=..., args_schema=...)` pattern to build custom tools:

```python
from loongflow.agentsdk.tools import FunctionTool
from pydantic import BaseModel, Field

class WriteToolArgs(BaseModel):
    file_path: str = Field(..., description="File path")
    content: str = Field(..., description="File content")

def build_custom_write_tool(context: Context, candidate_path: str) -> FunctionTool:
    """Build custom write tool with path validation"""
    
    async def write_func(file_path: str, content: str) -> dict:
        # Custom path validation logic
        if not file_path.startswith(candidate_path):
            # Auto-correct path
            filename = os.path.basename(file_path)
            resolved_path = os.path.join(candidate_path, filename)
        
        # Execute write operation
        written_path = Workspace.write_executor_file(context, resolved_path, content)
        return {"path": written_path, "message": "File written successfully"}
    
    return FunctionTool(
        func=write_func,
        args_schema=WriteToolArgs,
        name="Write",
        description="Custom write tool with path security check"
    )

# Use in agent
toolkit.register_tool(build_custom_write_tool(context, candidate_path))
```

### Integrating Tools in Agents

```python
from loongflow.agentsdk.tools import Toolkit, ReadTool, WriteTool, LsTool
from loongflow.framework.base.agent_base import AgentBase

class MyAgent(AgentBase):
    def __init__(self):
        super().__init__()
        self.toolkit = self._build_toolkit()
    
    def _build_toolkit(self) -> Toolkit:
        toolkit = Toolkit()
        toolkit.register_tool(ReadTool())
        toolkit.register_tool(WriteTool()) 
        toolkit.register_tool(LsTool())
        return toolkit
    
    async def __call__(self, **kwargs):
        # Use tool to process task
        response = await self.toolkit.arun("Read", {
            "file_path": kwargs["file_path"]
        })
        # ToolResponse processing
        if response.err_msg:
            # Handle error
            pass
        else:
            # Handle success result
            data = response.content[0].data
```

## Custom Tool Development

### Complex Tools Based on Pydantic

```python
from pydantic import BaseModel, Field
from loongflow.agentsdk.tools import FunctionTool

class DataAnalysisArgs(BaseModel):
    dataset_path: str = Field(..., description="Dataset path")
    analysis_type: str = Field(..., description="Analysis type")
    parameters: dict = Field(default={}, description="Analysis parameters")

class DataAnalysisTool(FunctionTool):
    def __init__(self):
        super().__init__(
            args_schema=DataAnalysisArgs,
            name="data_analysis",
            description="Execute data analysis task"
        )
    
    async def arun(self, *, args, tool_context=None):
        validated_args = self._prepare_call_args(args, tool_context)
        # Implement analysis logic
        return ToolResponse(content=[...])
```

### Tool Response Standardization

```python
from loongflow.agentsdk.message import ContentElement, MimeType

def build_success_response(data, tool_name):
    return ToolResponse(
        content=[
            ContentElement(
                mime_type=MimeType.APPLICATION_JSON,
                data=data,
                metadata={"tool": tool_name, "success": True}
            )
        ]
    )

def build_error_response(error_message, tool_name):
    return ToolResponse(
        content=[
            ContentElement(
                mime_type=MimeType.TEXT_PLAIN,
                data=error_message,
                metadata={"tool": tool_name, "error": True}
            )
        ],
        err_msg=error_message
    )
```

## Best Practices

### Security Considerations

1.  **Path Validation**: Use absolute paths or explicit project-relative paths for all file operations.
2.  **Code Sandbox**: Code execution should happen in an isolated environment with timeout limits.
3.  **Input Validation**: Strictly validate tool parameters to prevent injection attacks.
4.  **Permission Control**: Minimize tool permissions as needed.

### Performance Optimization

```python
# Batch tool execution
async def batch_tool_execution(toolkit, operations):
    results = {}
    for op_name, params in operations.items():
        results[op_name] = await toolkit.arun(op_name, params)
    return results

# Tool result caching
class CachedTool(FunctionTool):
    def __init__(self, base_tool, cache_ttl=300):
        super().__init__()
        self.base_tool = base_tool
        self.cache = {}
        self.cache_ttl = cache_ttl
```

### Error Handling

```python
try:
    response = await toolkit.arun("Read", {"file_path": path})
    if response.err_msg:
        # Handle tool-level error
        logging.error(f"Tool error: {response.err_msg}")
    else:
        # Handle success result
        data = response.content[0].data
except Exception as e:
    # Handle framework-level error
    logging.error(f"Framework error: {e}")
```

## Troubleshooting

### Common Issues

**Tool Not Found Error**
- Check tool name spelling.
- Verify if the tool is correctly registered to the Toolkit.

**Permission Errors**
- Check if authentication configuration is correct.
- Verify file path permissions.

**Execution Timeout**
- Adjust tool timeout settings.
- Optimize tool execution logic.

**Parameter Validation Failure**
- Check if parameters conform to the Pydantic model definition.
- Verify if required parameters are provided.

### Debugging Tips

```python
# Enable verbose logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Check tool declaration
declarations = toolkit.get_declarations()
print("Available tools:", declarations)

# Verify tool parameters
tool = toolkit.get("tool_name")
if tool:
    declaration = tool.get_declaration()
    print("Tool schema:", declaration)
```

LoongFlow Tool Components provide powerful and secure infrastructure for evolutionary agents, supporting complex multi-step task execution and tool composition, making them a key technical component for building autonomous agents.