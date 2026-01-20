# 进化存储器

进化存储器管理进化算法中的解决方案状态，支持并行进化、种群管理和检查点功能。

## 核心功能

### 基础架构
```python
from loongflow.agentsdk.memory.evolution.in_memory import InMemory

memory = InMemory(
    num_islands=3,           # 岛屿数量
    population_size=100,     # 种群大小
    elite_archive_size=50,   # 精英存档大小
    migration_interval=10    # 迁移间隔
)
```

### 解决方案管理
- **解决方案表示**: 使用`Solution`数据类存储代码、分数、父代关系
- **岛屿分配**: 解决方案根据父代或轮询分配到不同岛屿
- **种群控制**: 自动维护种群大小，淘汰低分解决方案

### 进化算法支持
- **精英选择**: 维护精英存档，保留最优解决方案
- **迁移机制**: 岛屿间定期迁移优秀解决方案
- **多样性管理**: 基于特征的MAP-Elites算法

## 使用方法

### 添加解决方案
```python
solution = Solution(
    solution="def solve(): return 42",
    score=0.85,
    parent_id="parent_001"
)
solution_id = await memory.add_solution(solution)
```

### 检索信息
```python
# 获取最佳解决方案
best_solutions = memory.get_best_solutions(island_id=0, top_k=5)

# 采样解决方案用于交叉/变异
parent = memory.sample(island_id=0, exploration_rate=0.1)

# 查看内存状态
status = memory.memory_status()
```

### 检查点管理
```python
# 保存检查点
await memory.save_checkpoint("./checkpoints/", "iteration_50")

# 加载检查点
memory.load_checkpoint("./checkpoints/iteration_50/")
```

## 配置选项

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `num_islands` | 并行进化岛屿数量 | 3 |
| `population_size` | 最大解决方案数量 | 100 |
| `elite_archive_size` | 精英存档大小 | 50 |
| `migration_interval` | 迁移间隔（代数） | 10 |

## 最佳实践

1. **岛屿配置**: 复杂问题使用3-5个岛屿进行并行探索
2. **检查点频率**: 长时间运行任务每10-20次迭代保存检查点
3. **种群大小**: 根据问题复杂度调整，避免过大内存占用

更多配置选项详见[配置指南](configuration.md)。