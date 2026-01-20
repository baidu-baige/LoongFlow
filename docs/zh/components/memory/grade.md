# 等级存储器

等级存储器提供智能体对话历史的分层管理，基于STM（短期）、MTM（中期）、LTM（长期）三层架构，支持自动压缩和令牌管理。

## 三层记忆架构

### 短期记忆 (STM)
- 存储最近对话历史
- 快速访问，不持久化
- 默认使用内存存储

### 中期记忆 (MTM)
- 存储压缩后的对话摘要
- 支持LLM自动压缩
- 可在会话间持久化

### 长期记忆 (LTM)
- 存储重要事实和知识
- 永久性存储
- 支持文件持久化

## 核心功能

### 初始化
```python
from loongflow.agentsdk.memory.grade.memory import GradeMemory

grade_memory = GradeMemory.create_default(
    model=llm_model,
    config=MemoryConfig(
        token_threshold=65536,  # 令牌阈值
        auto_compress=True      # 自动压缩
    )
)
```

### 消息管理
```python
# 添加消息
await grade_memory.add(message)

# 获取所有记忆内容
context = await grade_memory.get_memory()

# 手动提交重要信息到长期记忆
await grade_memory.commit_to_ltm(important_message)
```

### 自动压缩
当令牌数超过阈值时，系统自动触发压缩：
1. 将STM和MTM中的消息合并
2. 使用LLM生成摘要
3. 清除会话记忆，保留压缩结果

## 存储后端

### 内存存储
```python
from loongflow.agentsdk.memory.grade.storage import InMemoryStorage
storage = InMemoryStorage()  # 适用于开发和测试
```

### 文件存储
```python
from loongflow.agentsdk.memory.grade.storage import FileStorage  
storage = FileStorage("./memory_data/")  # 适用于生产环境
```

## 压缩器

### LLM压缩器
使用语言模型智能压缩对话历史：
```python
from loongflow.agentsdk.memory.grade.compressor import LLMCompressor
compressor = LLMCompressor(model)
```

## 最佳实践

1. **令牌管理**: 根据模型上下文长度设置合理的`token_threshold`
2. **压缩策略**: 重要信息手动提交到LTM避免丢失
3. **存储选择**: 开发环境使用内存存储，生产环境使用文件存储

更多配置选项详见[配置指南](configuration.md)。