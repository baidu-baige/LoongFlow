# Model Components

The model component provides an integration and management system for LLMs (Large Language Models), based on LiteLLM to implement a unified interface for multiple providers and tool calling capabilities.

## Architecture Design

### Core Abstraction Layer
- **BaseLLMModel**: The abstract base class for all model implementations.
- **LiteLLMModel**: A concrete implementation based on LiteLLM, supporting mainstream model providers such as OpenAI, Anthropic, Google, etc.
- **Formatter Pattern**: Responsible for the conversion and parsing of request/response formats.

### Request/Response Processing
- **CompletionRequest**: Standardized model request format, including messages, tool configurations, etc.
- **CompletionResponse**: Unified response format, supporting streaming processing.
- **CompletionUsage**: Token usage statistics.

## Core Classes

### BaseLLMModel (`base_llm_model.py`)
The model abstract base class, defining unified interface specifications:

```python
from loongflow.agentsdk.models import BaseLLMModel
from loongflow.agentsdk.models.llm_request import CompletionRequest

class CustomModel(BaseLLMModel):
    async def generate(self, request: CompletionRequest, stream: bool = False):
        # Model generation implementation
        pass
```

### LiteLLMModel (`litellm_model.py`)
Concrete implementation based on LiteLLM:

```python
from loongflow.agentsdk.models import LiteLLMModel

# Create model instance from config
model = LiteLLMModel.from_config({
    "model": "gpt-4",
    "url": "https://api.openai.com/v1",
    "api_key": "your-api-key",
    "timeout": 600,  # Optional
    "model_provider": "openai"  # Optional
})

# Or instantiate directly
model = LiteLLMModel(
    model_name="gpt-4",
    base_url="https://api.openai.com/v1",
    api_key="your-api-key",
    timeout=600,  # Optional
    model_provider="openai"  # Optional
)
```

### Request and Response Classes

**CompletionRequest** (`llm_request.py`):
```python
from loongflow.agentsdk.models import CompletionRequest

request = CompletionRequest(
    messages=[...],  # List of Message objects
    temperature=0.7,
    tools=[...],     # Tool calling configuration
    tool_choice="auto"
)
```

**CompletionResponse** (`llm_response.py`):
```python
from loongflow.agentsdk.models import CompletionResponse
from loongflow.agentsdk.message import ContentElement, ToolCallElement, ThinkElement

response = CompletionResponse(
    id="response-id",
    content=[
        ContentElement(mime_type="text/plain", data="Response content"),
        ThinkElement(content="Thinking content"),
        ToolCallElement(target="Tool Name", arguments={})
    ],      # Supports a list of ContentElement/ToolCallElement/ThinkElement
    usage=CompletionUsage(...),
    finish_reason="stop"
)
```

## Formatter Subsystem

### BaseFormatter (`formatter/base_formatter.py`)
The formatter abstract base class, defining request/response conversion interfaces.

### LiteLLMFormatter (`formatter/litellm_formatter.py`)
A formatter dedicated to LiteLLM, responsible for:
- Conversion between LoongFlow message format ↔ LiteLLM API format
- Parsing and processing of tool calling parameters
- Support for multimodal content (text, image, audio)
- Handling of streaming responses

```python
from loongflow.agentsdk.models.formatter import LiteLLMFormatter

formatter = LiteLLMFormatter()
llm_kwargs = formatter.format_request(
    request=request, 
    model_name=model_name, 
    base_url=base_url, 
    api_key=api_key,
    stream=stream,  # Streaming flag
    timeout=600,    # Timeout setting
    model_provider=None  # Optional: Specify model provider
)
```

## Configuration

### Basic Configuration
```yaml
llm_config:
  model: "gpt-4o"                    # Model name
  url: "https://api.openai.com/v1"   # API endpoint
  api_key: "your-api-key"           # Authentication key
```

### Advanced Configuration
```yaml
llm_config:
  model: "gpt-4"
  url: "https://api.openai.com/v1" 
  api_key: "your-api-key"
  model_provider: "openai"          # Optional: Specify provider
  timeout: 600                      # Timeout setting
  temperature: 0.7                  # Generation parameter
  max_tokens: 2000
```

## Usage Examples

### Basic Model Call
```python
from loongflow.agentsdk.models import LiteLLMModel
from loongflow.agentsdk.message import Message
from loongflow.agentsdk.message import ContentElement
from loongflow.agentsdk.models.llm_request import CompletionRequest

# Create model instance
model = LiteLLMModel.from_config({
    "model": "gpt-4",
    "url": "https://api.openai.com/v1", 
    "api_key": "your-api-key"
})

# Prepare messages
messages = [
    Message(
        role="user",
        content=[ContentElement(mime_type="text/plain", data="Please explain Artificial Intelligence")]
    )
]

# Create request
request = CompletionRequest(messages=messages)

# Call model
async for response in model.generate(request):
    for element in response.content:
        if hasattr(element, 'data'):
            print(element.data)
```

### Tool Calling Integration
```python
# Define tools
tools = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Calculate mathematical expression",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string"}
                }
            }
        }
    }
]

# Request with tool calling
request = CompletionRequest(
    messages=messages,
    tools=tools,
    tool_choice="auto"
)

# Handle tool calling response
from loongflow.agentsdk.message import ToolCallElement
async for response in model.generate(request):
    for element in response.content:
        if isinstance(element, ToolCallElement):
            print(f"Calling tool: {element.target}")
            print(f"Arguments: {element.arguments}")
```

### Streaming Response Processing
```python
# Enable streaming response
request = CompletionRequest(messages=messages)
async for chunk in model.generate(request, stream=True):
    for element in chunk.content:
        if hasattr(element, 'data'):
            print(element.data, end="", flush=True)
    
    if chunk.finish_reason:
        print(f"\nFinish reason: {chunk.finish_reason}")
```

## Advanced Features

### Custom Model Provider Support
Supports multiple model providers via configuration:
- OpenAI (`openai`)
- Azure OpenAI (`azure_openai`)
- Anthropic (`anthropic`)
- Google AI Studio (`google_ai_studio`)
- DeepSeek (`deepseek`)
- Baidu Qianfan (`qianfan`)
- Moonshot AI (`moonshot`)

### Multimodal Content Support
```python
# Image content
image_element = ContentElement(
    mime_type="image/png", 
    data="data:image/png;base64,..."
)

# Audio content  
audio_element = ContentElement(
    mime_type="audio/wav",
    data="data:audio/wav;base64,..."
)
```

### Error Handling
```python
try:
    async for response in model.generate(request):
        # Handle normal response
        pass
except Exception as e:
    # Handle model call error
    print(f"Model call failed: {e}")
```

## Best Practices

### Model Selection Guide
1.  **Task Matching**: Choose appropriate models based on task complexity.
2.  **Cost Optimization**: Balance model performance and call costs.
3.  **Failure Handling**: Implement retry mechanisms and fallback strategies.

### Performance Optimization
```python
# Appropriate timeout setting
model = LiteLLMModel(
    model_name="gpt-4",
    base_url="https://api.openai.com/v1",
    api_key="your-api-key",
    timeout=300  # 5-minute timeout
)

# Batch processing optimization
# (Framework supports asynchronous concurrent calls)
```

## Troubleshooting

### Common Issues

**Authentication Failure**
- Verify the correctness of the API key and endpoint URL.
- Check network connections and firewall settings.

**Rate Limits**
- Implement appropriate retry and backoff strategies.
- Consider using multiple API keys for rotation.

**Response Format Errors**
- Verify that the request format meets the model's requirements.
- Check the correctness of tool calling parameters.
- Review the parsing logs of `LiteLLMFormatter`.

The model component provides powerful and flexible LLM integration capabilities for LoongFlow agents, supporting complex AI application scenarios and tool calling functions.