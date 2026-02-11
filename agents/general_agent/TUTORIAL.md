
---

## 📖 Complete Tutorial: Using General Agent

This tutorial will guide you through everything you need to know to use General Agent effectively.

### Step-by-Step Learning Path

#### 🎯 Level 1: Your First 30 Minutes (Beginner)

**Goal**: Understand the basics and run your first agent.

**Step 1: Run the TODO List Example** (10 minutes)
```bash
cd LoongFlow
source .venv/bin/activate
export ANTHROPIC_API_KEY="your-key-here"
export ANTHROPIC_BASE_URL="your-base-url-here"

# Run the simplest example
./run_general.sh 01_todo_list
```

**What you'll see**:
- Agent planning the implementation
- Code being generated iteratively
- Evolution improving the solution
- Final TODO app in `output-todo-list/task_*/iteration_*/executor/work_dir/`

**Step 2: Test Your Generated Code** (5 minutes)
```bash
cd output-todo-list/task_*/iteration_*/executor/work_dir/
python todo_app.py
```

Try adding TODOs, marking them complete, and restarting to see persistence work!

**Step 3: Understand the Output Structure** (5 minutes)
```
output-todo-list/
└── task_<timestamp>/
    └── iteration_<N>/
        ├── planner/          # How agent planned
        ├── executor/         # Generated code
        │   └── work_dir/     # 🎯 YOUR CODE IS HERE
        ├── evaluator/        # How agent tested
        └── summary/          # What agent learned
```

**✅ Checkpoint**: You've run your first agent and understand the output structure!

---

#### 🎓 Level 2: Next 45 Minutes (Intermediate)

**Goal**: Learn about skills and multi-file projects.

**Step 1: Run the File Processor Example** (15 minutes)
```bash
./run_general.sh 02_file_processor
```

**What's different**:
- Agent uses a custom skill (`file-processing`)
- Generates 5+ files instead of 1
- More complex project structure
- Professional code organization

**Step 2: Understand Custom Skills** (15 minutes)

Skills are knowledge packages that guide the agent. Look at:
```bash
cat .claude/skills/file-processing/SKILL.md
```

You'll see:
- Domain-specific best practices
- Code templates
- Common patterns
- Error handling guidelines

**Step 3: Test with Sample Data** (10 minutes)
```bash
cd output-file-processor/task_*/iteration_*/executor/work_dir/

# Test CSV analysis
python main.py ../../../../../../agents/general_agent/examples/02_file_processor/sample_data/sales_data.csv

# Try different output formats
python main.py sample.csv --output report.txt
python main.py sample.csv --output stats.json --format json
```

**Step 4: Create Your First Custom Skill** (5 minutes)

```bash
# Create skill directory
mkdir -p .claude/skills/my-skill

# Create skill definition
cat > .claude/skills/my-skill/SKILL.md << 'EOF'
---
name: "my-skill"
description: "My custom skill for..."
---

# My Custom Skill

## Purpose
[Describe when to use this skill]

## Best Practices
- Practice 1
- Practice 2

## Code Patterns
```python
# Example pattern
def my_pattern():
    pass
```

**✅ Checkpoint**: You understand skills and can create multi-file projects!

---

#### 🚀 Level 3: Next 60 Minutes (Advanced)

**Goal**: Build production-ready bug detection agents.

**Step 1: Try the Analysis Tools First** (15 minutes)

Before running the agent, see what the tools can do:

```bash
cd agents/general_agent/examples/03_bug_hunter

# Static analysis
python ../../../.claude/skills/code-analysis/scripts/static_analyzer.py sample_codebase/

# Security scanning
python ../../../.claude/skills/code-analysis/scripts/security_scanner.py sample_codebase/

# Performance profiling
python ../../../.claude/skills/code-analysis/scripts/performance_profiler.py sample_codebase/user_manager.py
```

**What you'll see**:
- Automated bug detection (12+ issues)
- CWE mappings (CWE-89 SQL Injection, etc.)
- Performance bottlenecks (O(n²) loops)
- Security vulnerabilities

**Step 2: Run the Bug Hunter** (20 minutes)
```bash
cd LoongFlow
./run_general.sh 03_bug_hunter
```

The agent will:
1. Run all analysis tools
2. Manually review the code
3. Combine automated + manual findings
4. Generate comprehensive bug report
5. Create fixed versions of all files

**Step 3: Review the Output** (15 minutes)
```bash
cd output-bug-hunter/task_*/iteration_*/executor/work_dir/

# Read the bug report
cat BUG_REPORT.md | less

# See the fixes
cat CHANGES.md

# Compare before/after
diff sample_codebase/user_manager.py fixed_code/user_manager.py
```

**Step 4: Learn Security Patterns** (10 minutes)

```bash
# Read OWASP Top 10 guide
cat .claude/skills/code-analysis/references/owasp_top10_python.md | less

# Study CWE patterns
cat .claude/skills/code-analysis/references/cwe_quick_reference.md | less
```

**✅ Checkpoint**: You can build production-ready analysis agents with professional tools!

---

### 🎯 Creating Your Own Task

Now you're ready to create custom tasks! Here's how:

#### Step 1: Create Task Directory

```bash
cd LoongFlow/agents/general_agent/examples
mkdir my_custom_task
cd my_custom_task
```

#### Step 2: Create task_config.yaml

```bash
cat > task_config.yaml << 'EOF'
# Output directory
workspace_path: "./output-my-task"

# LLM configuration
llm_config:
  model: "anthropic/claude-3-5-sonnet-20241022"
  # Or use environment variables:
  # ANTHROPIC_API_KEY and ANTHROPIC_BASE_URL

# Component configuration
planners:
  general_planner:
    skills: []  # Add skill names here, e.g., ["code-analysis"]
    permission_mode: "acceptEdits"
    max_turns: 30

executors:
  general_executor:
    skills: []
    permission_mode: "acceptEdits"
    max_turns: 30

summarizers:
  general_summarizer:
    skills: []
    permission_mode: "acceptEdits"
    max_turns: 20

# Evolution configuration
evolve:
  # Your task description
  task: |
    Create a Python script that [describe what you want].

    Requirements:
    1. [Requirement 1]
    2. [Requirement 2]
    3. [Requirement 3]

    The script should:
    - [Feature 1]
    - [Feature 2]
    - [Feature 3]

  planner_name: "general_planner"
  executor_name: "general_executor"
  summary_name: "general_summarizer"

  max_iterations: 30
  target_score: 0.85
  concurrency: 1

  evaluator:
    timeout: 600
    agent:
      skills: []
      permission_mode: "acceptEdits"
      max_turns: 20

  database:
    storage_type: "in_memory"
    num_islands: 1
    population_size: 30
    checkpoint_interval: 5
    sampling_weight_power: 2
EOF
```

#### Step 3: Run Your Task

```bash
cd LoongFlow
./run_general.sh my_custom_task
```

#### Step 4: Monitor and Iterate

```bash
# Watch logs in real-time
tail -f output-my-task/task_*/iteration_*/planner/*.log

# Check intermediate results
ls -la output-my-task/task_*/iteration_*/executor/work_dir/

# If not satisfied, adjust task_config.yaml and re-run
```

---

### 🔧 Advanced Techniques

#### Technique 1: Using Multiple Skills

```yaml
planners:
  general_planner:
    skills: ["file-processing", "code-analysis", "my-custom-skill"]
```

Skills are combined - agent gets knowledge from all of them!

#### Technique 2: Adjusting Evolution Parameters

```yaml
evolve:
  max_iterations: 50      # More iterations = more refinement
  target_score: 0.95      # Higher score = higher quality requirement
  concurrency: 3          # Parallel evolution (faster, uses more resources)
```

**When to increase iterations**:
- Complex tasks
- High quality requirements
- Agent not converging

**When to increase concurrency**:
- You have compute resources
- Want faster results
- Exploring solution space

#### Technique 3: Custom Evaluation

Create `eval_program.py` in your task directory:

```python
#!/usr/bin/env python3
"""
Custom evaluator for my task
"""
import os
import sys

def evaluate(solution_path: str) -> float:
    """
    Evaluate the solution

    Args:
        solution_path: Path to executor/work_dir/

    Returns:
        Score between 0.0 and 1.0
    """
    score = 0.0

    # Check if required files exist
    required_files = ['main.py', 'config.py']
    for f in required_files:
        if os.path.exists(os.path.join(solution_path, f)):
            score += 0.2

    # Run tests
    try:
        # Your test logic here
        result = run_tests(solution_path)
        if result.success:
            score += 0.6
    except Exception as e:
        print(f"Tests failed: {e}")

    return min(1.0, score)

if __name__ == "__main__":
    solution_path = sys.argv[1]
    score = evaluate(solution_path)
    print(f"Score: {score}")
```

#### Technique 4: Debugging

```bash
# Run with debug logging
./run_general.sh my_task --log-level DEBUG

# Check what agent is thinking
cat output-my-task/task_*/iteration_*/planner/conversation.json | jq .

# See evaluator output
cat output-my-task/task_*/iteration_*/evaluator/evaluation.log
```

---

### 💡 Best Practices

#### DO ✅

1. **Start Simple**: Begin with 01_todo_list, then progress
2. **Use Skills**: Leverage existing skills before creating new ones
3. **Iterate**: Run, check output, adjust config, re-run
4. **Be Specific**: Clear task descriptions = better results
5. **Check Outputs**: Always verify generated code works
6. **Learn from Summaries**: Read summary/ dir to understand what agent learned

#### DON'T ❌

1. **Don't Skip Tutorials**: They teach important concepts
2. **Don't Use Too Many Iterations**: Start with 20-30, increase if needed
3. **Don't Ignore Errors**: Check logs if something fails
4. **Don't Forget API Keys**: Set ANTHROPIC_API_KEY before running
5. **Don't Mix Concerns**: One task should do one thing well
6. **Don't Expect Perfection**: Agent may need multiple runs

---

### 🎓 Understanding the Output

Every run creates this structure:

```
output-<task>/
└── task_<timestamp>/              # Unique run identifier
    ├── iteration_1/
    │   ├── planner/
    │   │   └── best_plan.md      # The plan it made
    │   ├── executor/
    │   │   └── work_dir/         # Generated code
    │   ├── evaluator/
    │   │   └── score.txt         # How good is it? (0.0-1.0)
    │   └── summary/
    │       └── insights.md       # What agent learned
    ├── iteration_2/              # Improved version
    ├── iteration_3/              # Even better
    └── ...
```

**Key files to check**:
- `executor/work_dir/` - Your generated code
- `evaluator/score.txt` - Quality score
- `summary/insights.md` - Lessons learned
- `planner/plan.md` - Implementation strategy

**Finding the best iteration**:
```bash
# Find iteration with highest score
grep -r "score" output-my-task/task_*/iteration_*/evaluator/ | sort -k2 -n
```

---

### 🚨 Troubleshooting

#### Issue: "API Key Not Found"
```bash
# Check if set
echo $ANTHROPIC_API_KEY

# Set it
export ANTHROPIC_API_KEY="your-key"

# Or in task_config.yaml
llm_config:
  api_key: "your-key"
```

#### Issue: "No Code Generated"
- Check `planner/best_plan.md` to see what went wrong
- Increase `max_turns` in config
- Make task description more specific

#### Issue: "Score Always Low"
- Task might be too complex
- Add custom `eval_program.py`
- Check if agent is understanding requirements

#### Issue: "Agent Takes Too Long"
- Reduce `max_iterations`
- Simplify task
- Use faster model (like `deepseek-v3.2`)

#### Issue: "Skill Not Found"
```bash
# Check skill exists
ls -la .claude/skills/your-skill/SKILL.md

# Verify path in config
grep "skills:" task_config.yaml
```

---

### 📚 Next Steps

After completing this tutorial:

1. **Create Custom Skills**
   - Build skills for your domain
   - Share them with your team
   - Contribute to LoongFlow

2. **Integrate into Workflow**
   - Add to CI/CD pipelines
   - Create pre-commit hooks
   - Build custom agents for your team

3. **Join the Community**
   - Share your experiences
   - Report issues on GitHub
   - Contribute improvements

---

### 🎯 Quick Reference

**Running Tasks**:
```bash
./run_general.sh <task_name>                    # Run task
./run_general.sh <task_name> --background       # Run in background
./run_general.sh stop <task_name>               # Stop background task
./run_general.sh <task_name> --log-level DEBUG  # Debug mode
```

**Common Paths**:
- Task configs: `agents/general_agent/examples/<task>/task_config.yaml`
- Skills: `.claude/skills/<skill>/SKILL.md`
- Output: `output-<task>/task_*/iteration_*/executor/work_dir/`
- Logs: `output-<task>/task_*/iteration_*/planner/*.log`

**Configuration Keys**:
- `workspace_path`: Where to save output
- `max_iterations`: How many evolution cycles
- `target_score`: When to stop (0.0-1.0)
- `concurrency`: Parallel evolutions
- `skills`: Which skills to load
- `max_turns`: Max conversation turns

---

**🎉 You're now ready to use General Agent effectively!**

Start with [01_todo_list](examples/01_todo_list/), progress through all 4 examples, and then build your own custom agents!
