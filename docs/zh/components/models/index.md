# 模型组件

模型组件提供LLM（大语言模型）的集成和管理系统，基于LiteLLM实现多供应商的统一接口和工具调用功能。

## 架构设计

### 核心抽象层
- **BaseLLMModel**: 所有模型实现的抽象基类
- **LiteLLMModel**: 基于LiteLLM的具体实现，支持OpenAI、Anthropic、Google等主流模型提供商
- **Formatter模式**: 负责请求/响应格式的转换和解析

### 请求/响应处理
- **CompletionRequest**: 标准化的模型请求格式，包含消息、工具配置等
- **CompletionResponse**: 统一的响应格式，支持流式处理
- **CompletionUsage**: Token使用统计信息

## 核心类

### BaseLLMModel (`base_llm_model.py`)
模型抽象基类，定义统一的接口规范：

```python
from loongflow.agentsdk.models import BaseLLMModel
from loongflow.agentsdk.models.llm_request import CompletionRequest

class CustomModel(BaseLLMModel):
    async def generate(self, request: CompletionRequest, stream: bool = False):
        # 模型生成实现
        pass
```

### LiteLLMModel (`litellm_model.py`)
基于LiteLLM的具体实现：

```python
from loongflow.agentsdk.models import LiteLLMModel

# 从配置创建模型实例
model = LiteLLMModel.from_config({
    "model": "gpt-4",
    "url": "https://api.openai.com/v1",
    "api_key": "your-api-key",
    "timeout": 600,  # 可选参数
    "model_provider": "openai"  # 可选参数
})

# 或直接实例化
model = LiteLLMModel(
    model_name="gpt-4",
    base_url="https://api.openai.com/v1",
    api_key="your-api-key",
    timeout=600,  # 可选参数
    model_provider="openai"  # 可选参数
)
```

### 请求与响应类

**CompletionRequest** (`llm_request.py`):
```python
from loongflow.agentsdk.models import CompletionRequest

request = CompletionRequest(
    messages=[...],  # Message对象列表
    temperature=0.7,
    tools=[...],     # 工具调用配置
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
        ContentElement(mime_type="text/plain", data="响应内容"),
        ThinkElement(content="思考内容"),
        ToolCallElement(target="工具名称", arguments={})
    ],      # 支持ContentElement/ToolCallElement/ThinkElement列表
    usage=CompletionUsage(...),
    finish_reason="stop"
)
```

## Formatter子系统

### BaseFormatter (`formatter/base_formatter.py`)
格式化器抽象基类，定义请求/响应转换接口。

### LiteLLMFormatter (`formatter/litellm_formatter.py`)
LiteLLM专用的格式化器，负责：
- LoongFlow消息格式 ↔ LiteLLM API格式的转换
- 工具调用参数的解析和处理
- 多模态内容（文本、图像、音频）的支持
- 流式响应的处理

```python
from loongflow.agentsdk.models.formatter import LiteLLMFormatter

formatter = LiteLLMFormatter()
llm_kwargs = formatter.format_request(
    request=request, 
    model_name=model_name, 
    base_url=base_url, 
    api_key=api_key,
    stream=stream,  # 流式处理标志
    timeout=600,    # 超时设置
    model_provider=None  # 可选：指定模型提供商
)
```

## 配置

### 基础配置
```yaml
llm_config:
  model: "gpt-4o"                    # 模型名称
  url: "https://api.openai.com/v1"   # API端点
  api_key: "your-api-key"           # 认证密钥
```

### 高级配置
```yaml
llm_config:
  model: "gpt-4"
  url: "https://api.openai.com/v1" 
  api_key: "your-api-key"
  model_provider: "openai"          # 可选：指定提供商
  timeout: 600                      # 超时设置
  temperature: 0.7                  # 生成参数
  max_tokens: 2000
```

## 使用示例

### 基础模型调用
```python
from loongflow.agentsdk.models import LiteLLMModel
from loongflow.agentsdk.message import Message
from loongflow.agentsdk.message import ContentElement
from loongflow.agentsdk.models.llm_request import CompletionRequest

# 创建模型实例
model = LiteLLMModel.from_config({
    "model": "gpt-4",
    "url": "https://api.openai.com/v1", 
    "api_key": "your-api-key"
})

# 准备消息
messages = [
    Message(
        role="user",
        content=[ContentElement(mime_type="text/plain", data="请解释人工智能")]
    )
]

# 创建请求
request = CompletionRequest(messages=messages)

# 调用模型
async for response in model.generate(request):
    for element in response.content:
        if hasattr(element, 'data'):
            print(element.data)
```

### 工具调用集成
```python
# 定义工具
tools = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "计算数学表达式",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string"}
                }
            }
        }
    }
]

# 带工具调用的请求
request = CompletionRequest(
    messages=messages,
    tools=tools,
    tool_choice="auto"
)

# 处理工具调用响应
from loongflow.agentsdk.message import ToolCallElement
async for response in model.generate(request):
    for element in response.content:
        if isinstance(element, ToolCallElement):
            print(f"调用工具: {element.target}")
            print(f"参数: {element.arguments}")
```

### 流式响应处理
```python
# 启用流式响应
request = CompletionRequest(messages=messages)
async for chunk in model.generate(request, stream=True):
    for element in chunk.content:
        if hasattr(element, 'data'):
            print(element.data, end="", flush=True)
    
    if chunk.finish_reason:
        print(f"\n完成原因: {chunk.finish_reason}")
```

## 高级功能

### 自定义模型提供商支持
通过配置支持多种模型提供商：
- OpenAI (`openai`)
- Azure OpenAI (`azure_openai`) 
- Anthropic (`anthropic`)
- Google AI Studio (`google_ai_studio`)
- DeepSeek (`deepseek`)
- 百度千帆 (`qianfan`)
- 月之暗面 (`moonshot`)

### 多模态内容支持
```python
# 图像内容
image_element = ContentElement(
    mime_type="image/png", 
    data="data:image/png;base64,..."
)

# 音频内容  
audio_element = ContentElement(
    mime_type="audio/wav",
    data="data:audio/wav;base64,..."
)
```

### 错误处理
```python
try:
    async for response in model.generate(request):
        # 处理正常响应
        pass
except Exception as e:
    # 处理模型调用错误
    print(f"模型调用失败: {e}")
```

## 最佳实践

### 模型选择指南
1. **任务匹配**: 根据任务复杂度选择合适的模型
2. **成本优化**: 平衡模型性能和调用成本
3. **失败处理**: 实现重试机制和降级策略

### 性能优化
```python
# 合适的超时设置
model = LiteLLMModel(
    model_name="gpt-4",
    base_url="https://api.openai.com/v1",
    api_key="your-api-key",
    timeout=300  # 5分钟超时
)

# 批量处理优化
# （框架支持异步并发调用）
```

## 故障排除

### 常见问题

**认证失败**
- 验证API密钥和端点URL的正确性
- 检查网络连接和防火墙设置

**速率限制**  
- 实现适当的重试和退避策略
- 考虑使用多个API密钥轮换

**响应格式错误**
- 验证请求格式符合模型要求
- 检查工具调用参数的正确性
- 查看LiteLLMFormatter的解析日志

模型组件为LoongFlow智能体提供强大而灵活的LLM集成能力，支持复杂的AI应用场景和工具调用功能。