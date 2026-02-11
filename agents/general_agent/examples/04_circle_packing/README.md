# 🎯 Example 04: Circle Packing Optimization

**Difficulty**: ⭐⭐⭐⭐ Expert
**Time**: 20-30 minutes
**Goal**: Solve complex geometric optimization using evolutionary algorithms

---

## 🎯 What You'll Learn

This expert-level example demonstrates:

- ✅ **Advanced Optimization**: Solving NP-hard geometric problems
- ✅ **Custom Evaluation**: Writing sophisticated eval_program.py
- ✅ **Constraint Handling**: Managing complex geometric constraints
- ✅ **Algorithm Design**: Implementing search heuristics
- ✅ **Performance Tuning**: 1000-second runtime limits

**Prerequisites**: Complete Examples [01](../01_todo_list/), [02](../02_file_processor/), and [03](../03_bug_hunter/)

---

## 🧩 The Problem

**Goal**: Pack `n` non-overlapping circles into the unit square [0,1] × [0,1] such that the **sum of their radii is maximized**.

### Constraints

1. **Circles must fit**: All circles must be entirely inside [0,1] × [0,1]
2. **No overlaps**: Circles must be disjoint (not touching)
3. **Valid output**: Must return `(centers, radii, sum_radii)`

### Why This is Hard

- **NP-hard problem**: No known polynomial-time solution
- **Combinatorial explosion**: ~10^20 possible configurations for n=10
- **Trade-offs**: Fewer large circles vs. many small circles
- **Edge effects**: Square boundaries make optimal packing non-trivial

### Real-World Applications

- Warehouse layout optimization
- Satellite constellation design
- Molecular structure prediction
- Network coverage planning
- PCB component placement

---

## 🚀 Quick Start

```bash
# 1. Navigate to LoongFlow root
cd /path/to/LoongFlow

# 2. Run the example (this will take longer!)
./run_general.sh 04_circle_packing

# 3. Monitor progress (in another terminal)
tail -f agents/general_agent/examples/04_circle_packing/run.log

# 4. Check generated solutions
cd output-circle-packing/task_*/iteration_*/executor/work_dir/
python -c "from run_packing import run_packing; c, r, s = run_packing(5); print(f'Sum of radii: {s}')"
```

**Note**: This example uses `max_iterations: 200` because optimization is iterative!

---

## 📦 What the Agent Generates

A Python module implementing:

### run_packing(num_circles)
Main function that returns:
- `centers`: NumPy array of shape (n, 2) with (x, y) coordinates
- `radii`: Array of n non-negative radii
- `sum_radii`: Total sum of all radii

### Common Approaches

Agents typically try:

1. **Greedy Placement**
   ```python
   # Place circles one by one, maximizing each radius
   for i in range(num_circles):
       find_largest_valid_position()
   ```

2. **Grid-Based**
   ```python
   # Divide square into grid, place circles in cells
   grid_size = int(np.sqrt(num_circles))
   max_radius = 0.5 / grid_size
   ```

3. **Random Search + Refinement**
   ```python
   # Generate random configurations, keep best
   for _ in range(1000):
       config = generate_random_packing()
       if score(config) > best_score:
           best = config
   ```

4. **Hexagonal Patterns**
   ```python
   # Use known optimal patterns for infinite plane
   # Adjust for square boundary
   ```

---

## 🧪 Custom Evaluation

This example includes [eval_program.py](eval_program.py) - a custom evaluator!

### What It Does

```python
def evaluate(solution_path: str) -> float:
    \"\"\"
    Evaluates circle packing solution

    Returns:
        Score between 0.0 and 1.0 based on sum_radii
    \"\"\"
    # Import the generated module
    # Call run_packing(5), run_packing(10), run_packing(20)
    # Verify constraints (no overlaps, circles inside square)
    # Score based on sum_radii relative to theoretical maximum
```

### Scoring Logic

```
score = min(1.0, sum_radii / theoretical_maximum)

Where theoretical_maximum is estimated from known bounds
```

### Validation Checks

1. **Circles inside square**: `0 ≤ x-r, x+r, y-r, y+r ≤ 1`
2. **No overlaps**: `distance(i, j) ≥ radii[i] + radii[j]` for all pairs
3. **Valid numbers**: No NaN, Inf, or negative radii
4. **Correct shape**: `centers.shape == (num_circles, 2)`

**If validation fails → score = 0.0**

---

## 🎯 Expected Results

### Iteration 1-3 (Score: 0.2-0.4)
- Basic grid placement
- Small radii, lots of wasted space
- sum_radii ≈ 0.5-1.0 for n=10

### Iteration 3-5 (Score: 0.5-0.7)
- Smarter placement strategies
- Better space utilization
- sum_radii ≈ 1.5-2.0 for n=10

### Iteration 5+ (Score: 0.8-0.9) ✅
- Near-optimal solutions
- Advanced algorithms (simulated annealing, genetic algorithms)
- sum_radii ≈ 2.5+ for n=10

### Theoretical Bounds

For n circles in unit square:
- **Lower bound**: ~sqrt(n) * 0.2 (naive grid)
- **Upper bound**: ~sqrt(n * π/(2*sqrt(3))) (hexagonal packing)
- **Realistic**: 60-80% of upper bound

---

## ⚙️ Configuration Deep Dive

Key settings in [task_config.yaml](task_config.yaml):

```yaml
evolve:
  task: |
    Write a search function to pack num_circles disjoint disks
    in [0,1]×[0,1] to maximize sum of radii.

    Key geometric insights:
    - Hexagonal patterns have maximum density π/(2√3) ≈ 0.9069
    - Edge effects make square packing harder
    - Varied radii can utilize space better than uniform

  max_iterations: 200              # ⚠️ Much higher!
  target_score: 1.0                # Perfect score (very hard)

  evaluator:
    timeout: 1200                  # 20 minutes per evaluation

  database:
    population_size: 90            # Larger population for diversity
    checkpoint_interval: 1         # Save more frequently
```

**Why 200 iterations?**
- Optimization problems need many refinement cycles
- Each iteration explores different algorithmic approaches
- Convergence is gradual, not sudden

---

## 🧠 Key Geometric Insights

The task prompt provides hints to guide the agent:

### Insight 1: Hexagonal Packing
```
Maximum density for infinite plane: π/(2*sqrt(3)) ≈ 0.9069
```

Optimal packing in infinite plane follows hexagonal pattern:
```
  o   o   o
   o   o   o
  o   o   o
```

### Insight 2: Edge Effects
```
Square container makes packing harder than infinite packing
```

Circles near edges have fewer neighbors, reducing efficiency.

### Insight 3: Radius Variation
```
Similar radius circles form regular patterns,
varied radii allow better space utilization
```

Strategy: Use large circles in center, small ones to fill gaps.

### Insight 4: Symmetry Breaking
```
Perfect symmetry may not yield optimal packing
```

Sometimes asymmetric configurations pack better!

---

## 🛠️ Testing Your Solution

```bash
cd output-circle-packing/task_*/iteration_*/executor/work_dir/

# Test with different numbers of circles
python -c "
from run_packing import run_packing
import numpy as np

for n in [3, 5, 10, 15, 20]:
    centers, radii, sum_r = run_packing(n)
    print(f'n={n:2d}: sum_radii={sum_r:.4f}, avg_radius={sum_r/n:.4f}')

    # Verify no overlaps
    for i in range(n):
        for j in range(i+1, n):
            dist = np.sqrt(np.sum((centers[i] - centers[j])**2))
            if radii[i] + radii[j] > dist:
                print(f'  ⚠️ Overlap detected between circles {i} and {j}!')
"
```

### Visualize Results

```python
import matplotlib.pyplot as plt
from run_packing import run_packing

centers, radii, sum_r = run_packing(10)

fig, ax = plt.subplots(figsize=(8,8))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_aspect('equal')

for (x, y), r in zip(centers, radii):
    circle = plt.Circle((x, y), r, fill=False, edgecolor='blue')
    ax.add_patch(circle)

ax.set_title(f'Circle Packing: n=10, sum_radii={sum_r:.4f}')
plt.savefig('packing.png')
print('Saved visualization to packing.png')
```

---

## 🎓 Advanced Topics

### Multi-Island Evolution

This example uses `num_islands: 1`, but you can enable multiple islands:

```yaml
database:
  num_islands: 5                   # 5 independent populations
  population_size: 90              # 90 solutions per island
```

**Benefit**: Different islands explore different approaches in parallel!

### Adaptive Sampling

```yaml
database:
  sampling_weight_power: 2         # Strongly favor high-scoring solutions
```

- Power = 1: Linear sampling (more exploration)
- Power = 2: Quadratic sampling (more exploitation)
- Power = 3: Cubic sampling (very greedy)

### Checkpointing

```yaml
database:
  checkpoint_interval: 1           # Save after every iteration
```

Checkpoints saved to: `output-circle-packing/database/checkpoints/`

Can resume from checkpoint if interrupted!

---

## 🏆 Challenge Goals

### Bronze (Easy): sum_radii > 2.0 for n=10
Basic approach with reasonable space utilization.

### Silver (Medium): sum_radii > 2.5 for n=10
Smart placement strategy, good space efficiency.

### Gold (Hard): sum_radii > 2.8 for n=10
Near-optimal algorithm, advanced techniques.

### Platinum (Expert): sum_radii > 3.0 for n=10
State-of-the-art performance, publication-worthy!

---

## 🛠️ Common Issues & Solutions

### Issue: "Evaluation timeout"

```yaml
evaluator:
  timeout: 2400  # Increase to 40 minutes if needed
```

Complex algorithms may need more time.

### Issue: "Score stuck at 0.3"

- Agent might be using naive grid placement
- Check iteration logs to see what algorithms were tried
- Increase max_iterations to explore more approaches
- Add hints about hexagonal packing to task description

### Issue: "Memory error"

```yaml
database:
  population_size: 30  # Reduce from 90
```

Large populations use more memory.

### Issue: "Overlapping circles"

- Check constraint validation in eval_program.py
- Agent's algorithm might have floating-point errors
- Add stricter validation with epsilon tolerance

---

## 🎉 Next Steps

Congratulations on completing all 4 examples! Now you can:

1. **Create your own optimization task**
   - Modify this example for different constraints
   - Try packing rectangles, ellipses, or 3D spheres

2. **Experiment with evolution parameters**
   ```yaml
   concurrency: 5                    # Run 5 parallel evolutions
   num_islands: 10                   # 10 independent populations
   max_iterations: 500               # Really push the limits!
   ```

3. **Integrate with research/production**
   - Use General Agent for hyperparameter optimization
   - Solve real-world layout/packing problems
   - Benchmark against published algorithms

4. **Build your own examples**
   - See [TUTORIAL.md](../../TUTORIAL.md) for creating custom tasks
   - Share your examples with the community!

---

## 📚 Key Takeaways

- ✅ **Custom evaluation** enables complex scoring logic
- ✅ **Geometric constraints** can be validated programmatically
- ✅ **High iteration counts** are normal for optimization
- ✅ **Evolution parameters** significantly affect convergence
- ✅ **Real-world problems** can be solved with General Agent

**You've completed the entire quickstart series!** 🎉

Check out the [LoongFlow documentation](../../../../README.md) for more advanced features like multi-island evolution, custom memory systems, and production deployments!

---

## 📖 Further Reading

- **Circle Packing Wikipedia**: https://en.wikipedia.org/wiki/Circle_packing
- **Known Optimal Packings**: http://hydra.nat.uni-magdeburg.de/packing/csq/csq.html
- **Hexagonal Packing**: https://en.wikipedia.org/wiki/Circle_packing_in_a_square

---

**Happy optimizing!** 🚀🎯
