# 消息组件

消息组件是 LoongFlow 框架中用于智能体间通信的核心模块，提供了标准化的消息格式和内容元素系统。

## 核心概念

### 消息角色 (Role)
- **system**: 系统消息，用于提供指令或上下文
- **user**: 用户消息，表示用户输入
- **assistant**: 助手消息，表示智能体响应
- **tool**: 工具消息，表示工具调用结果

### MIME 类型支持
- **text/plain**: 纯文本内容
- **application/json**: JSON 数据
- **image/jpeg**, **image/png**: 图像内容
- **audio/mpeg**: 音频内容
- **video/mp4**: 视频内容

## 核心类

### Message 类
基础消息类，包含完整的消息信息：

```python
from loongflow.agentsdk.message import Message, Role, ContentElement, MimeType

# 创建基础文本消息
message = Message(
    role=Role.USER,
    content=[ContentElement(mime_type=MimeType.TEXT_PLAIN, data="你好，世界！")],
    sender="user123",
    trace_id="trace-001",
    conversation_id="conv-456"
)
```

### 内容元素 (Elements)

#### ContentElement
通用内容元素，支持各种媒体类型：

```python
from loongflow.agentsdk.message import ContentElement, MimeType

# 文本内容
text_element = ContentElement(
    mime_type=MimeType.TEXT_PLAIN, 
    data="这是一段文本内容"
)

# JSON 内容
json_element = ContentElement(
    mime_type=MimeType.APPLICATION_JSON,
    data={"key": "value", "count": 42}
)
```

#### ToolCallElement
工具调用元素，用于请求执行工具：

```python
from loongflow.agentsdk.message import ToolCallElement
import uuid

tool_call = ToolCallElement(
    call_id=uuid.uuid4(),
    target="weather_api",
    arguments={"city": "北京", "unit": "celsius"}
)
```

#### ToolOutputElement
工具执行结果元素：

```python
from loongflow.agentsdk.message import ToolOutputElement, ContentElement, ToolStatus
import uuid

tool_output = ToolOutputElement(
    call_id=uuid.uuid4(),  # 必须与对应的 ToolCallElement 的 call_id 一致
    tool_name="calculator",
    status=ToolStatus.SUCCESS,
    result=[ContentElement(mime_type=MimeType.TEXT_PLAIN, data="计算结果: 42")]
)
```

#### ThinkElement
智能体思考过程元素：

```python
from loongflow.agentsdk.message import ThinkElement

think_element = ThinkElement(
    content="我需要分析用户的问题并制定解决策略..."
)
```

#### EvolveResultElement
进化进程结果元素（用于进化算法场景）：

```python
from loongflow.agentsdk.message import EvolveResultElement

evolve_result = EvolveResultElement(
    best_score=0.95,
    best_solution="优化后的算法代码",
    evaluation="在测试集上表现良好",
    start_time="2024-01-01 10:00:00",
    end_time="2024-01-01 11:00:00",
    cost_time=3600,
    last_iteration=50,
    total_iterations=100
)
```

## 便捷构造方法

Message 类提供了多种便捷的构造方法：

```python
from loongflow.agentsdk.message import Message, Role

# 从文本创建消息
text_message = Message.from_text("你好，这是一个文本消息")

# 从内容创建消息
json_message = Message.from_content(
    data={"result": "success"}, 
    mime_type="application/json"
)

# 从工具调用创建消息
tool_message = Message.from_tool_call(
    target="search_engine",
    arguments={"query": "人工智能"}
)

# 从工具执行结果创建消息
tool_output_message = Message.from_tool_output(
    call_id=call_id,           # 必须的 call_id 参数
    tool_name="search_engine", # 必须的 tool_name 参数
    status="success",
    result=[ContentElement(mime_type="text/plain", data="搜索结果")],
    sender="search_service"
)

# 从思考内容创建消息
think_message = Message.from_think("我正在思考解决方案...")

# 从元素列表创建消息
elements_message = Message.from_elements([
    ContentElement(mime_type="text/plain", data="第一部分"),
    ThinkElement(content="思考过程")
])

# 从媒体内容创建消息（专用方法）
media_message = Message.from_media(
    sender="camera_sensor",
    mime_type="image/jpeg",
    data=image_data,
    role=Role.TOOL
)
```

## 消息操作

### 序列化与反序列化
```python
# 序列化为字典
message_dict = message.to_dict()

# 从字典创建消息
new_message = Message.from_dict(message_dict)
```

### 元素过滤
```python
# 获取消息中的所有内容元素
content_elements = message.get_elements(ContentElement)

# 获取思考元素
think_elements = message.get_elements(ThinkElement)

# 获取进化结果元素
evolve_elements = message.get_elements(EvolveResultElement)
```

## 使用示例

### 完整的对话流程
```python
from loongflow.agentsdk.message import Message, Role, ContentElement, ToolCallElement, ToolOutputElement
import uuid

# 用户提问
user_message = Message.from_text(
    "北京今天的天气怎么样？", 
    role=Role.USER,
    sender="user123"
)

# 智能体请求工具调用
call_id = uuid.uuid4()
agent_message = Message.from_tool_call(
    target="weather_service",
    arguments={"city": "北京"},
    sender="weather_agent",
    role=Role.ASSISTANT
)

# 工具执行结果
tool_message = Message.from_tool_output(
    call_id=call_id,
    tool_name="weather_service",  # 必须指定工具名称
    status="success",
    result=[ContentElement(mime_type="text/plain", data="北京: 晴朗, 25°C")],
    sender="weather_service",
    role=Role.TOOL
)
```

### 包含思考过程的消息
```python
from loongflow.agentsdk.message import Message, ThinkElement, ContentElement

# 智能体的思考+响应消息
reasoning_message = Message.from_elements([
    ThinkElement(content="用户想知道天气信息，我需要调用天气服务"),
    ContentElement(mime_type="text/plain", data="让我查询一下北京的天气...")
], sender="assistant", role=Role.ASSISTANT)
```

### 进化算法结果消息
```python
from loongflow.agentsdk.message import Message, EvolveResultElement

# 进化算法执行结果
evolve_message = Message.from_elements([
    EvolveResultElement(
        best_score=0.98,
        best_solution="最终的数学公式",
        evaluation="在验证集上达到了98%的准确率",
        start_time="10:00:00",
        end_time="11:30:00",
        cost_time=5400,
        last_iteration=25,
        total_iterations=100
    )
], sender="evolve_agent", role=Role.ASSISTANT)
```

## 最佳实践

1. **使用合适的角色**: 根据消息来源选择正确的 Role 枚举值
2. **结构化的内容**: 使用适当的元素类型来组织消息内容
3. **包含追踪信息**: 使用 trace_id 和 conversation_id 来追踪消息流
4. **元数据利用**: 在 metadata 字段中存储额外的上下文信息
5. **正确的工具调用匹配**: 确保 ToolOutputElement 的 call_id 与对应的 ToolCallElement 一致

消息组件为 LoongFlow 框架提供了强大而灵活的消息传递能力，支持复杂的智能体协作场景。