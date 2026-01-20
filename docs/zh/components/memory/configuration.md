# 存储器配置指南

LoongFlow内存系统的完整配置参考，包括进化存储器和等级存储器。

## 进化存储器配置

### 基础配置
```python
from loongflow.agentsdk.memory.evolution.in_memory import InMemory

memory = InMemory(
    num_islands=3,           # 岛屿数量
    population_size=100,     # 每个岛屿最大解决方案数
    elite_archive_size=50,   # 精英存档大小
    migration_interval=10,   # 迁移间隔
    output_path="./output"   # 检查点输出路径
)
```

### 高级参数
```python
memory = InMemory(
    # 进化算法参数
    boltzmann_temperature=1.0,      # 采样温度
    migration_rate=0.2,             # 迁移比例
    use_sampling_weight=True,       # 使用采样权重
    sampling_weight_power=1.0,      # 权重指数
    
    # 多样性管理
    feature_dimensions=["complexity", "diversity", "score"],
    feature_bins=10,                # 特征分箱数
    feature_scaling_method="minmax" # 特征缩放方法
)
```

## 等级存储器配置

### 基础配置
```python
from loongflow.agentsdk.memory.grade.memory import GradeMemory, MemoryConfig

grade_memory = GradeMemory.create_default(
    model=llm_model,
    config=MemoryConfig(
        token_threshold=65536,  # 自动压缩的令牌阈值
        auto_compress=True      # 启用自动压缩
    )
)
```

### 自定义存储后端
```python
from loongflow.agentsdk.memory.grade.storage import FileStorage
from loongflow.agentsdk.memory.grade.compressor import LLMCompressor

# 自定义存储和压缩器
stm_storage = FileStorage("./stm_data/")
mtm_storage = FileStorage("./mtm_data/") 
ltm_storage = FileStorage("./ltm_data/")
compressor = LLMCompressor(model, custom_prompt="请压缩以下对话历史")

grade_memory = GradeMemory(
    stm=ShortTermMemory(stm_storage),
    mtm=MediumTermMemory(mtm_storage, compressor),
    ltm=LongTermMemory(ltm_storage),
    token_counter=token_counter,
    config=MemoryConfig(token_threshold=32768)
)
```

## YAML配置文件

### 进化存储器配置
```yaml
evolution:
  num_islands: 3
  population_size: 100
  elite_archive_size: 50
  migration_interval: 10
  migration_rate: 0.2
  output_path: "./evolution_output"
```

### 等级存储器配置
```yaml
grade:
  token_threshold: 65536
  auto_compress: true
  storage:
    stm:
      type: "in_memory"
    mtm: 
      type: "file"
      path: "./mtm_data"
    ltm:
      type: "file" 
      path: "./ltm_data"
```

## 环境变量配置

```bash
# 进化存储器
export EVOLUTION_NUM_ISLANDS=3
export EVOLUTION_POPULATION_SIZE=100
export EVOLUTION_OUTPUT_PATH="./output"

# 等级存储器  
export GRADE_TOKEN_THRESHOLD=65536
export GRADE_AUTO_COMPRESS=true
```

## 性能调优

### 内存优化
```python
# 减小种群大小和岛屿数量
memory = InMemory(
    num_islands=2,           # 减少岛屿数
    population_size=50,      # 减小种群
    elite_archive_size=25    # 精简精英存档
)
```

### 压缩优化
```python
# 调整压缩阈值
config = MemoryConfig(
    token_threshold=32768,  # 降低阈值，更频繁压缩
    auto_compress=True
)
```

## 故障排除

### 调试模式
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### 内存监控
```python
# 查看进化存储器状态
status = memory.memory_status()
print(f"当前种群: {status['global_status']['total_valid_solutions']}")

# 查看等级存储器大小  
size = await grade_memory.get_size()
print(f"总消息数: {size}")
```

## 最佳实践

1. **开发环境**: 使用默认配置，重点关注功能实现
2. **生产环境**: 根据任务复杂度调整岛屿数量和种群大小
3. **大规模部署**: 考虑Redis等分布式存储后端

更多实现细节参考[进化存储器](evolution.md)和[等级存储器](grade.md)。