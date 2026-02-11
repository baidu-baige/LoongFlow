# 📝 Example 01: TODO List Application

**Difficulty**: ⭐ Beginner
**Time**: 5-10 minutes
**Goal**: Build your first AI-generated command-line TODO app

---

## 🎯 What You'll Learn

This is the perfect starting point for General Agent! In this example, you'll learn:

- ✅ How the **Plan-Execute-Summary (PES)** workflow works
- ✅ How General Agent evolves code through iterations
- ✅ How to run a task and find your generated code
- ✅ Basic task configuration in `task_config.yaml`

---

## 🚀 Quick Start (3 Commands)

```bash
# 1. Navigate to LoongFlow root
cd /path/to/LoongFlow

# 2. Set your API keys (if not already set)
export ANTHROPIC_API_KEY="your-key"
export ANTHROPIC_BASE_URL="your-endpoint"

# 3. Run the example
./run_general.sh 01_todo_list
```

That's it! The agent will now:
1. **Plan** how to build a TODO app
2. **Execute** the implementation
3. **Evaluate** the solution
4. **Summarize** what it learned
5. Repeat until it creates a working app!

---

## 📂 What This Example Generates

The General Agent will create a command-line TODO application with:

- ✅ **Add TODOs**: Add new tasks with descriptions
- ✅ **Mark Complete**: Mark tasks as done
- ✅ **List TODOs**: View all tasks with status
- ✅ **Persistent Storage**: Saves to JSON file
- ✅ **Clean Interface**: User-friendly command-line interaction

---

## 🔍 Where to Find Your Code

After running, check the output directory:

```bash
cd output-todo-list/task_*/iteration_*/executor/work_dir/
```

You'll find files like:
- `todo_app.py` - Main application
- `todos.json` - Data storage file (created on first run)

---

## 🧪 Test Your Generated App

```bash
# Navigate to the generated code
cd output-todo-list/task_*/iteration_*/executor/work_dir/

# Run the TODO app
python todo_app.py

# Try these commands:
# - Add a todo: "Buy groceries"
# - Add another: "Write documentation"
# - List all todos
# - Mark one as complete
# - List again to see the change
# - Exit and restart to verify persistence works!
```

---

## 📊 Understanding the Output Structure

```
output-todo-list/
└── task_<timestamp>/              # Unique run ID
    └── iteration_1/               # First iteration
        ├── planner/
        │   └── best_plan.md       # The agent's plan
        ├── executor/
        │   └── work_dir/          # 🎯 YOUR CODE IS HERE
        │       ├── todo_app.py
        │       └── todos.json
        ├── evaluator/
        │   └── score.txt          # Quality score (0.0-1.0)
        └── summary/
            └── insights.md        # What the agent learned

    └── iteration_2/               # Improved version (if score < target)
    └── iteration_3/               # Even better...
    └── ...
```

**Key Points**:
- Each iteration tries to improve the solution
- Higher `score.txt` = better solution
- Agent stops when `score >= target_score` (default: 0.85)

---

## ⚙️ Configuration Explained

Let's look at [task_config.yaml](task_config.yaml):

```yaml
workspace_path: "./output-todo-list"     # Where to save output

llm_config:
  model: "anthropic/deepseek-v3.2"       # Which model to use

evolve:
  task: |
    Create a command-line TODO list application...  # Task description

  max_iterations: 30                     # Max evolution cycles
  target_score: 0.85                     # Stop when score ≥ 0.85
  concurrency: 1                         # Number of parallel evolutions
```

**What Each Part Does**:
- `workspace_path`: Where General Agent saves all output
- `llm_config`: Which LLM to use (Anthropic-compatible models)
- `task`: Natural language description of what to build
- `max_iterations`: Safety limit (stops even if target not reached)
- `target_score`: Quality threshold (0.0 = bad, 1.0 = perfect)

---

## 🎓 What Happens Behind the Scenes

### Iteration 1:
1. **Planner**: "I need to create a TODO app with add, list, and mark functions"
2. **Executor**: Writes `todo_app.py` with basic structure
3. **Evaluator**: Runs the code, tests features, scores it (e.g., 0.6)
4. **Summarizer**: "The app works but needs better error handling and persistence"

### Iteration 2:
1. **Planner**: Uses feedback from iteration 1 to improve
2. **Executor**: Adds JSON storage and error handling
3. **Evaluator**: Tests again, scores it (e.g., 0.87) ✅
4. **Summary**: "Successfully implemented all features!"

**Evolution stops** because score (0.87) ≥ target (0.85)!

---

## 🎯 Expected Results

A successful run should produce:

1. **Scores improving** across iterations:
   - Iteration 1: 0.4-0.6 (basic functionality)
   - Iteration 2: 0.7-0.8 (added features)
   - Iteration 3: 0.85+ (complete solution) ✅

2. **Working TODO app** that:
   - Doesn't crash on invalid input
   - Saves data between sessions
   - Has clear user interface
   - Implements all required features

---

## 🛠️ Common Issues & Solutions

### Issue: "API Key Not Found"
```bash
# Solution: Set environment variables
export ANTHROPIC_API_KEY="your-key-here"
export ANTHROPIC_BASE_URL="your-endpoint"

# Or add to task_config.yaml:
llm_config:
  api_key: "your-key"
  url: "your-endpoint"
```

### Issue: "No code generated"
- Check `planner/best_plan.md` - did planning succeed?
- Increase `max_turns: 30` in config
- Check logs: `agents/general_agent/examples/01_todo_list/run.log`

### Issue: "Score always low"
- Check `evaluator/*.log` to see why tests fail
- Task might be unclear - make description more specific
- Increase `max_iterations` to give more chances

### Issue: "Can't find output"
```bash
# Find the latest output
ls -lt output-todo-list/task_*/iteration_* | head
cd output-todo-list/task_<tab>/iteration_<tab>/executor/work_dir/
```

---

## 🎉 Next Steps

Congratulations on running your first General Agent example! Now try:

1. **Modify the task**: Edit `task_config.yaml` and add new requirements
   ```yaml
   task: |
     Create a TODO app that also supports:
     - Priority levels (High, Medium, Low)
     - Due dates
     - Categories/tags
   ```

2. **Try the next example**: [02_file_processor](../02_file_processor/) - Learn about skills and multi-file projects

3. **Read the tutorial**: [TUTORIAL.md](../../TUTORIAL.md) - Comprehensive guide covering all examples

---

## 📚 Key Takeaways

- ✅ General Agent uses **Plan-Execute-Summary** cycles to evolve code
- ✅ Each iteration improves on previous attempts
- ✅ Generated code is in `executor/work_dir/`
- ✅ Evolution stops when `score >= target_score`
- ✅ You can customize behavior via `task_config.yaml`

**Ready to learn more?** Head to [Example 02: File Processor](../02_file_processor/) to discover the power of custom skills! 🚀
