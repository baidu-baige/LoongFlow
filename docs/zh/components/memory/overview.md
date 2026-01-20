# 存储器组件

LoongFlow的内存系统为智能体提供分层存储管理，包括进化算法的状态管理和对话历史的管理。

## 核心架构

内存系统分为两个主要部分：

### 进化存储器 (`src/loongflow/agentsdk/memory/evolution/`)
- 管理进化算法中解决方案的存储和种群管理
- 支持岛屿模式的并行进化算法
- 提供检查点功能，支持保存和恢复进化进度

### 等级存储器 (`src/loongflow/agentsdk/memory/grade/`)
- 基于STM（短期记忆）、MTM（中期记忆）、LTM（长期记忆）三级存储
- 支持消息历史自动压缩
- 管理智能体对话上下文和历史

## 主要特性

- **进化状态管理**: 岛屿分配、种群管理、精英存档
- **历史压缩**: 基于LLM的智能消息压缩
- **持久化存储**: 支持内存和文件存储后端
- **自动管理**: 令牌计数、自动压缩、内存清理

## 快速开始

```python
from loongflow.agentsdk.memory.evolution.in_memory import InMemory
from loongflow.agentsdk.memory.grade.memory import GradeMemory

# 进化存储器
evolution_memory = InMemory(num_islands=3, population_size=100)

# 等级存储器
grade_memory = GradeMemory.create_default(model)
```

继续阅读以下文档了解详细信息：
- [进化存储器](evolution.md) - 进化状态管理
- [等级存储器](grade.md) - 对话历史管理
- [配置指南](configuration.md) - 配置选项