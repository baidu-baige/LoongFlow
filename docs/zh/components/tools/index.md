# 工具组件

LoongFlow 工具组件为智能体提供安全、可扩展的工具调用框架，支持文件操作、代码执行、Shell 命令等常用功能，是构建进化式智能体的核心基础设施。

## 架构设计

### 核心类层级

```
BaseTool (抽象基类)
    ├── FunctionTool (函数工具基类)
    │   ├── ReadTool (文件读取工具)
    │   ├── WriteTool (文件写入工具)  
    │   ├── LsTool (目录列表工具)
    │   ├── ShellTool (Shell 命令执行工具)
    │   ├── ExecuteCodeTool (代码执行工具)
    │   ├── AgentTool (智能体调用工具)
    │   ├── TodoReadTool (待办读取工具)
    │   ├── TodoWriteTool (待办写入工具)
    │   └── 自定义工具
    └── Toolkit (工具包管理器)
```

### 核心概念

- **ToolContext**: 工具执行上下文，包含认证状态和运行时信息
- **ToolResponse**: 标准化的工具执行响应格式
- **FunctionDeclaration**: 工具功能声明，用于 LLM 函数调用

## 核心类详解

### BaseTool - 工具抽象基类

```python
from loongflow.agentsdk.tools import BaseTool
from loongflow.agentsdk.tools.tool_context import ToolContext
from loongflow.agentsdk.tools.tool_response import ToolResponse

class CustomTool(BaseTool):
    """自定义工具示例"""
    
    def __init__(self, *, name, description):
        super().__init__(name=name, description=description)
    
    def get_declaration(self) -> Optional[FunctionDeclarationDict]:
        """返回工具的功能声明"""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "input": {"type": "string", "description": "输入参数"}
                },
                "required": ["input"]
            }
        }
    
    async def arun(self, *, args: Dict[str, Any], tool_context: Optional[ToolContext] = None) -> ToolResponse:
        """异步执行工具"""
        # 实现工具逻辑
        return ToolResponse(content=[...])
```

### FunctionTool - 函数工具实现

基于 Pydantic 模型的工具实现，支持参数验证：

```python
from pydantic import BaseModel, Field
from loongflow.agentsdk.tools import FunctionTool

class CalculatorArgs(BaseModel):
    expression: str = Field(..., description="数学表达式")
    
class CalculatorTool(FunctionTool):
    def __init__(self):
        super().__init__(
            args_schema=CalculatorArgs,
            name="calculator",
            description="执行数学计算"
        )
    
    def run(self, *, args, tool_context=None):
        validated_args = self._prepare_call_args(args, tool_context)
        result = eval(validated_args["expression"])
        return ToolResponse(content=[...])
```

### Toolkit - 工具包管理器

```python
from loongflow.agentsdk.tools import Toolkit, ReadTool, WriteTool

# 创建工具包
toolkit = Toolkit()

# 注册工具（支持认证配置）
toolkit.register_tool(ReadTool())
toolkit.register_tool(WriteTool())

# 获取工具声明（返回OpenAI函数调用格式）
declarations = toolkit.get_declarations()

# 同步执行工具
response = toolkit.run("Read", args={"file_path": "/path/to/file"})

# 异步执行工具
response = await toolkit.arun("Read", args={"file_path": "/path/to/file"})
```

## 内置工具

### ReadTool - 文件读取工具

工具名称：**`Read`**

```python
from loongflow.agentsdk.tools import ReadTool

read_tool = ReadTool()
response = await read_tool.arun({
    "file_path": "/absolute/path/to/file.py",
    "offset": 10,      # 可选：起始行号
    "limit": 20        # 可选：读取行数
})

# 返回格式: ToolResponse
# - 文本文件: 返回格式化文本内容（带行号）
# - 图片文件: 返回 {"type": "image", "path": file_path}
# - PDF文件: 返回 {"type": "pdf", "path": file_path}
# - Notebook: 返回 {"type": "notebook", "path": file_path}

# 支持的文件类型：
# - 文本文件 (.txt, .py, .json, .csv, .md, .html, .xml)
# - 图片文件 (.png, .jpg, .jpeg, .gif) 
# - PDF 文件 (.pdf)
# - Jupyter Notebook (.ipynb)
```

### WriteTool - 文件写入工具

工具名称：**`Write`**

```python
from loongflow.agentsdk.tools import WriteTool

write_tool = WriteTool()
response = await write_tool.arun({
    "file_path": "relative/or/absolute/path.txt",
    "content": "文件内容"
})

# 特性：
# - 自动创建目录结构
# - 支持相对路径（相对于项目根目录，解析到项目根）
# - 覆盖已存在文件
# - 返回格式: ToolResponse 包含成功消息
```

### TodoReadTool - 待办读取工具

工具名称：**`TodoRead`**

```python
from loongflow.agentsdk.tools import TodoReadTool

read_todo = TodoReadTool()
response = await read_todo.arun({})

# 返回结构 (在 ToolResponse.content[0].data 中):
# {
#   "message": "Todo list retrieved successfully.",
#   "file_path": "./todo_list.json",
#   "todos": [
#     {
#       "id": "uuid",
#       "content": "任务描述", 
#       "status": "pending"
#     }
#   ]
# }

# 特性：
# - 自动从ToolContext获取文件路径或使用默认路径
# - 支持自定义todo文件存储位置
# - 无参数调用，简单易用
```

### TodoWriteTool - 待办写入工具

工具名称：**`TodoWrite`**

```python
from loongflow.agentsdk.tools import TodoWriteTool

write_tool = TodoWriteTool()
response = await write_tool.arun({
    "todos": [
        {
            "content": "完成任务A",
            "status": "completed"
        },
        {
            "content": "开始任务B", 
            "status": "in_progress"
        }
    ]
})

# 返回结构 (在 ToolResponse.content[0].data 中):
# {
#   "message": "Todo list updated successfully.",
#   "file_path": "./todo_list.json",
#   "todos": [...]
# }

# 特性：
# - 支持自定义todo项和状态管理
# - 自动创建不存在的目录结构
# - 与TodoReadTool共享文件路径配置
```

### LsTool - 目录列表工具

工具名称：**`LS`**

```python
from loongflow.agentsdk.tools import LsTool

ls_tool = LsTool()
response = await ls_tool.arun({
    "path": "/absolute/path/to/directory",
    "ignore": ["*.pyc", "__pycache__"]  # 可选：忽略模式
})

# 返回结构 (在 ToolResponse.content[0].data 中):
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

# 特性：
# - 支持路径自动转换（相对路径转绝对）
# - 支持忽略模式过滤
# - 只遍历一层目录
```

### ShellTool - Shell 命令执行工具

工具名称：**`ShellTool`**

```python
from loongflow.agentsdk.tools import ShellTool

shell_tool = ShellTool()
response = await shell_tool.arun({
    "commands": [
        {"command": "ls -la", "dir": "/working/directory"},
        {"command": "python script.py"}
    ]
})

# 返回每个命令的结构：
# {
#   "command": "执行的命令",
#   "dir": "工作目录",
#   "returncode": 退出码,
#   "stdout": "标准输出",
#   "stderr": "标准错误"
# }

# 特性：
# - 支持异步和同步执行
# - 批量命令执行
# - 错误处理和结果返回
```

### ExecuteCodeTool - 代码执行工具

工具名称：**`ExecuteCode`**

```python
from loongflow.agentsdk.tools import ExecuteCodeTool

code_tool = ExecuteCodeTool()

# 执行内联代码
response = await code_tool.arun({
    "mode": "code",
    "code": "print('Hello World')",
    "language": "python",  # 可选，默认为"python"
    "timeout": 30
})

# 执行文件
response = await code_tool.arun({
    "mode": "file", 
    "file_path": "/path/to/script.py",
    "language": "python",
    "timeout": 60
})

# 返回结构 (在 ToolResponse.content[0].data 中):
# {
#   "stdout": "标准输出",
#   "stderr": "标准错误", 
#   "returncode": 退出码,
#   "error": "错误信息",
#   "execution_time": 执行时间(秒)
# }

# 特性：
# - 支持Python代码和脚本文件
# - 超时控制
# - 子进程隔离执行
```

### AgentTool - 智能体调用工具

工具名称：**继承自传入智能体的 `name` 属性**

将 `AgentBase` 智能体封装为可调用工具，支持智能体之间的协作调用：

```python
from loongflow.agentsdk.tools import AgentTool
from loongflow.agentsdk.tools.tool_response import ToolResponse
from loongflow.agentsdk.message.message import Message, ContentElement, MimeType
from pydantic import BaseModel, Field
from loongflow.framework.base.agent_base import AgentBase

# 定义智能体输入模式
class MyAgentInput(BaseModel):
    request: str = Field(..., description="用户请求文本")

# 创建智能体
class MyAgent(AgentBase):
    name = "my_agent"
    description = "示例智能体"
    input_schema = MyAgentInput
    
    async def run(self, request: str) -> Message:
        """智能体主要逻辑实现"""
        # 处理请求并返回消息
        response_text = f"处理结果: {request}"
        return Message.from_text(
            data=response_text,
            sender=self.name,
            role="assistant",
            mime_type=MimeType.TEXT_PLAIN,
        )
    
    async def interrupt_impl(self) -> Message:
        """中断处理逻辑"""
        return Message.from_text("智能体已中断")

# 创建AgentTool包装
agent = MyAgent()
agent_tool = AgentTool(agent)

# 在其他智能体中调用（正确方式）
response = await agent_tool.arun(args={
    "request": "处理这个任务的详细描述"
}, tool_context=None)

# 或者使用同步调用
response = agent_tool.run(args={
    "request": "处理这个任务的详细描述"
}, tool_context=None)

# 返回的ToolResponse包含从Message转换的ContentElement
if response.content:
    content = response.content[0]
    print(f"响应类型: {content.mime_type}")
    print(f"响应数据: {content.data}")

# 特性：
# - **工具名称**: 继承自智能体的 `name` 属性
# - **参数模式**: 基于智能体的 `input_schema`，若无则使用默认的 `{"request": "string"}` 模式
# - **消息转换**: 智能体返回的Message被转换为ToolResponse，提取ContentElement内容
# - **调用模式**: 支持同步 `run()` 和异步 `arun()` 调用
# - **声明生成**: 自动生成JSON Schema用于LLM函数调用
```

### AgentTool 功能声明详解

AgentTool 自动生成功能声明，示例结构：

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

### 高级用法：消息处理机制

AgentTool 的 `_wrap_message_as_response()` 方法智能处理消息转换：

- **ContentElement提取**: 优先从Message中提取所有ContentElement
- **后备机制**: 若无ContentElement，则将整个Message序列化为JSON
- **消息类型支持**: 支持文本、JSON、图像等多种MIME类型

```python
# 自定义消息转换逻辑示例
message = await agent(**args)
response = AgentTool._wrap_message_as_response(message)
print(f"转换后内容元素数量: {len(response.content)}")
```

## 认证和上下文管理

### ToolContext 认证管理

```python
from loongflow.agentsdk.tools.tool_context import ToolContext, AuthConfig, AuthCredential, AuthType

# 创建认证配置
auth_config = AuthConfig(scheme=AuthType.API_KEY, key="openai_api")

# 创建认证凭证  
credential = AuthCredential(auth_type=AuthType.API_KEY, api_key="sk-...")

# 在工具上下文中设置认证
context = ToolContext(function_call_id="call_123")
context.set_auth(auth_config, credential)

# 工具执行时使用认证上下文
response = await tool.arun(args={}, tool_context=context)
```

### Toolkit 认证集成和管理方法

```python
from loongflow.agentsdk.tools import Toolkit, AuthConfig, AuthCredential, AuthType

# 创建工具包
toolkit = Toolkit()

# 1. 注册工具时添加认证（推荐方式）
toolkit.register_tool(
    api_tool, 
    auths=[(auth_config, credential)]
)

# 2. 单独设置认证（已注册工具）
toolkit.set_auth("tool_name", auth_config, credential)

# 3. 获取工具认证
stored_cred = toolkit.get_auth("tool_name", auth_config)

# 4. 获取工具实例
tool = toolkit.get("tool_name")

# 5. 列出所有已注册工具
tool_names = toolkit.list_tools()

# 6. 注销工具
toolkit.unregister_tool("tool_name")

# 7. 获取工具上下文
context = toolkit.get_context("tool_name")

# 8. 获取所有工具声明（返回OpenAI函数调用格式的列表）
# 格式：[{"type": "function", "function": {...tool_declaration...}}, ...]
declarations = toolkit.get_declarations()
```

### 高级用法：上下文管理和执行控制

```python
from loongflow.agentsdk.tools import Toolkit, ToolContext

# 使用自定义上下文执行工具
context = ToolContext(function_call_id="custom_session", state={"custom_data": "value"})
response = await toolkit.arun("tool_name", args={"param": "value"}, tool_context=context)

# 确保工具上下文存在（内部方法）
tool_context = toolkit.ensure_context("tool_name", external_context=context)

# 批量工具执行
async def execute_tools_sequentially(toolkit, operations: list):
    results = []
    for tool_name, args_dict in operations:
        result = await toolkit.arun(tool_name, args=args_dict)
        results.append((tool_name, result))
        if result.err_msg:
            break  # 出错时停止
    return results

# 错误处理和工具链
try:
    operations = [
        ("LS", {"path": "./"}),
        ("Read", {"file_path": "./README.md"}),
        ("ExecuteCode", {"mode": "code", "code": "print('test')"})
    ]
    results = await execute_tools_sequentially(toolkit, operations)
except Exception as e:
    print(f"执行工具链失败: {e}")
```

## 实际使用模式

### 项目中的工具构建模式

LoongFlow 项目实际使用 `FunctionTool(func=..., args_schema=...)` 模式构建自定义工具：

```python
from loongflow.agentsdk.tools import FunctionTool
from pydantic import BaseModel, Field

class WriteToolArgs(BaseModel):
    file_path: str = Field(..., description="文件路径")
    content: str = Field(..., description="文件内容")

def build_custom_write_tool(context: Context, candidate_path: str) -> FunctionTool:
    """构建带有路径验证的自定义写入工具"""
    
    async def write_func(file_path: str, content: str) -> dict:
        # 自定义路径验证逻辑
        if not file_path.startswith(candidate_path):
            # 自动修正路径
            filename = os.path.basename(file_path)
            resolved_path = os.path.join(candidate_path, filename)
        
        # 执行写入操作
        written_path = Workspace.write_executor_file(context, resolved_path, content)
        return {"path": written_path, "message": "File written successfully"}
    
    return FunctionTool(
        func=write_func,
        args_schema=WriteToolArgs,
        name="Write",
        description="自定义写入工具，包含路径安全性检查"
    )

# 在智能体中使用
toolkit.register_tool(build_custom_write_tool(context, candidate_path))
```

### 在智能体中集成工具

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
        # 使用工具处理任务
        response = await self.toolkit.arun("Read", {
            "file_path": kwargs["file_path"]
        })
        # ToolResponse 处理
        if response.err_msg:
            # 处理错误
            pass
        else:
            # 处理成功结果
            data = response.content[0].data
```


## 自定义工具开发

### 基于 Pydantic 的复杂工具

```python
from pydantic import BaseModel, Field
from loongflow.agentsdk.tools import FunctionTool

class DataAnalysisArgs(BaseModel):
    dataset_path: str = Field(..., description="数据集路径")
    analysis_type: str = Field(..., description="分析类型")
    parameters: dict = Field(default={}, description="分析参数")

class DataAnalysisTool(FunctionTool):
    def __init__(self):
        super().__init__(
            args_schema=DataAnalysisArgs,
            name="data_analysis",
            description="执行数据分析任务"
        )
    
    async def arun(self, *, args, tool_context=None):
        validated_args = self._prepare_call_args(args, tool_context)
        # 实现分析逻辑
        return ToolResponse(content=[...])
```

### 工具响应标准化

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

## 最佳实践

### 安全性考虑

1. **路径验证**: 所有文件操作使用绝对路径或明确的项目相对路径
2. **代码沙箱**: 代码执行在隔离环境进行，设置超时限制
3. **输入验证**: 严格验证工具参数，防止注入攻击
4. **权限控制**: 根据需要最小化工具权限

### 性能优化

```python
# 批量工具执行
async def batch_tool_execution(toolkit, operations):
    results = {}
    for op_name, params in operations.items():
        results[op_name] = await toolkit.arun(op_name, params)
    return results

# 工具结果缓存
class CachedTool(FunctionTool):
    def __init__(self, base_tool, cache_ttl=300):
        super().__init__()
        self.base_tool = base_tool
        self.cache = {}
        self.cache_ttl = cache_ttl
```

### 错误处理

```python
try:
    response = await toolkit.arun("Read", {"file_path": path})
    if response.err_msg:
        # 处理工具级别错误
        logging.error(f"Tool error: {response.err_msg}")
    else:
        # 处理成功结果
        data = response.content[0].data
except Exception as e:
    # 处理框架级别错误
    logging.error(f"Framework error: {e}")
```

## 故障排除

### 常见问题

**工具未找到错误**
- 检查工具名称拼写
- 验证工具是否正确注册到 Toolkit

**权限错误**  
- 检查认证配置是否正确
- 验证文件路径权限

**执行超时**
- 调整工具超时设置
- 优化工具执行逻辑

**参数验证失败**
- 检查参数是否符合 Pydantic 模型定义
- 验证必需参数是否提供

### 调试技巧

```python
# 启用详细日志
import logging
logging.basicConfig(level=logging.DEBUG)

# 检查工具声明
declarations = toolkit.get_declarations()
print("Available tools:", declarations)

# 验证工具参数
tool = toolkit.get("tool_name")
if tool:
    declaration = tool.get_declaration()
    print("Tool schema:", declaration)
```

LoongFlow 工具组件为进化式智能体提供了强大而安全的基础设施，支持复杂的多步骤任务执行和工具组合，是构建自主智能体的关键技术组件。