# General Agent Visualizer Design

## 🎯 Overview

A web-based dashboard for monitoring General Agent's Plan-Execute-Summary (PES) workflow in real-time.

## 🎨 Design Philosophy

Unlike Math Agent's evolution tree visualizer (focused on optimization), General Agent visualizer focuses on:
- **Task Progress Tracking**: Monitor PES phases in real-time
- **Code Generation Preview**: See generated files as they're created
- **Quality Metrics**: Track scores and improvements across iterations
- **Multi-file Project Support**: View entire project structures
- **Skill Usage Insights**: See which skills were applied

## 📊 Dashboard Layout

```
┌─────────────────────────────────────────────────────────────┐
│  General Agent Dashboard                   [Stop] [Refresh] │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Task: 01_todo_list              Status: Running ●  │   │
│  │  Iteration: 3 / 30               Score: 0.75 / 0.85 │   │
│  │  Time: 5m 23s                    Phase: Executing   │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                               │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  📈 Score History                                    │   │
│  │  1.0 ┤                                              │   │
│  │  0.8 ┤          ╭─●                                 │   │
│  │  0.6 ┤    ●─●─●─╯                                   │   │
│  │  0.4 ┤  ●─╯                                         │   │
│  │  0.2 ┤●─╯                                           │   │
│  │  0.0 └───────────────────────────────────────────► │   │
│  │       1   2   3   4   5   6   7   8   9   10       │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                               │
│  ┌──────────────────┬──────────────────────────────────┐   │
│  │  📂 Iterations   │  📄 Current Iteration Details    │   │
│  ├──────────────────┤                                  │   │
│  │ ✅ Iteration 1   │  Phase: Execute                   │   │
│  │    Score: 0.45   │  ────────────────────────────────│   │
│  │                  │                                  │   │
│  │ ✅ Iteration 2   │  📋 Plan Summary:                │   │
│  │    Score: 0.67   │  - Add persistent storage        │   │
│  │                  │  - Implement mark complete       │   │
│  │ ▶️ Iteration 3   │  - Improve error handling        │   │
│  │    Score: 0.75   │                                  │   │
│  │    (Running)     │  📁 Generated Files (3):         │   │
│  │                  │  ├─ todo_app.py         [View]   │   │
│  │ ⏳ Iteration 4   │  ├─ todos.json          [View]   │   │
│  │    (Pending)     │  └─ README.md           [View]   │   │
│  │                  │                                  │   │
│  │                  │  ✓ Evaluator Result:             │   │
│  │                  │    Score: 0.75                   │   │
│  │                  │    Passed: 7/10 tests            │   │
│  │                  │                                  │   │
│  │                  │  💡 Summary Insights:            │   │
│  │                  │  - Storage works correctly       │   │
│  │                  │  - Need better input validation  │   │
│  └──────────────────┴──────────────────────────────────┘   │
│                                                               │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  🔍 Code Viewer                                      │   │
│  │  File: todo_app.py                         [Close]  │   │
│  │  ────────────────────────────────────────────────── │   │
│  │  1  #!/usr/bin/env python3                         │   │
│  │  2  """Simple TODO list application"""             │   │
│  │  3                                                  │   │
│  │  4  import json                                     │   │
│  │  5  from pathlib import Path                        │   │
│  │  6                                                  │   │
│  │  7  class TodoManager:                              │   │
│  │  8      def __init__(self, filepath="todos.json"): │   │
│  │  9          self.filepath = Path(filepath)         │   │
│  │ 10          self.todos = self.load()               │   │
│  │ ...                                                 │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## 🧩 Key Features

### 1. Real-time Progress Monitor
- **Live Status**: Running / Completed / Failed
- **Current Phase**: Planning / Executing / Evaluating / Summarizing
- **Progress Bar**: Visual iteration progress
- **Estimated Time**: Based on historical data

### 2. Score Tracking
- **Line Chart**: Score evolution across iterations
- **Target Line**: Show target_score threshold
- **Best Score Marker**: Highlight best iteration
- **Score Breakdown**: Individual test scores

### 3. Iteration Browser
- **Left Panel**: List of all iterations with icons
  - ✅ Completed with high score
  - ⚠️ Completed with low score
  - ▶️ Currently running
  - ⏳ Pending
- **Right Panel**: Detailed view of selected iteration
  - Plan summary
  - Generated files list
  - Evaluation results
  - Summary insights

### 4. Multi-file Code Viewer
- **File Tree**: Collapsible directory structure
- **Syntax Highlighting**: Python, JSON, YAML, Markdown
- **Line Numbers**: Easy code reference
- **Diff View**: Compare with previous iteration
- **Download**: Export generated code

### 5. Skills & Tools Insights
- **Skills Used**: Show which skills were loaded
- **Tool Calls**: Timeline of tool usage (Read, Write, Edit, Bash)
- **Error Log**: Display any errors encountered
- **Token Usage**: Track LLM token consumption

### 6. Task Comparison
- **Multiple Tasks**: Compare different tasks side-by-side
- **Best Solutions**: Highlight best iteration from each task
- **Export Results**: Download reports

## 🔧 Technical Architecture

### Backend (Flask)
```python
# agents/general_agent/visualizer/visualizer.py

class GeneralAgentService:
    def list_tasks(self) -> List[str]:
        """List all tasks in output directories"""

    def get_task_overview(self, task_id: str) -> Dict:
        """Get task metadata and current status"""

    def get_iteration_details(self, task_id: str, iteration: int) -> Dict:
        """Get plan, execution, evaluation, summary for an iteration"""

    def get_file_content(self, task_id: str, iteration: int, filepath: str) -> str:
        """Get content of a generated file"""

    def get_score_history(self, task_id: str) -> List[Dict]:
        """Get score progression"""

    def get_live_status(self, task_id: str) -> Dict:
        """Get current execution status (for active tasks)"""
```

### Frontend (HTML/CSS/JS)
```
static/
├── index.html          # Main dashboard
├── css/
│   ├── main.css        # Layout and theme
│   └── syntax.css      # Code syntax highlighting
└── js/
    ├── app.js          # Main application logic
    ├── chart.js        # Score history chart
    └── codeviewer.js   # File viewer component
```

### Data Sources
```
output-<task>/
└── task_<timestamp>/
    ├── metadata.json              # NEW: Task-level metadata
    ├── iteration_1/
    │   ├── planner/
    │   │   ├── best_plan.md
    │   │   └── metadata.json      # NEW: Phase metadata
    │   ├── executor/
    │   │   ├── work_dir/          # Generated files
    │   │   └── metadata.json
    │   ├── evaluator/
    │   │   ├── score.txt
    │   │   └── metadata.json
    │   └── summary/
    │       ├── insights.md
    │       └── metadata.json
    └── iteration_2/
        └── ...
```

## 📝 Implementation Checklist

### Phase 1: Basic Dashboard (MVP)
- [ ] Flask server setup
- [ ] Task list endpoint
- [ ] Basic HTML layout
- [ ] Score history chart
- [ ] File browser

### Phase 2: Real-time Updates
- [ ] WebSocket for live updates
- [ ] Auto-refresh on iteration completion
- [ ] Progress indicators

### Phase 3: Advanced Features
- [ ] Code diff viewer
- [ ] Multi-task comparison
- [ ] Export functionality
- [ ] Skills usage analytics

### Phase 4: Polish
- [ ] Dark/Light theme
- [ ] Responsive design
- [ ] Keyboard shortcuts
- [ ] Search and filter

## 🎯 Usage Examples

### Start Visualizer
```bash
# From LoongFlow root
python agents/general_agent/visualizer/visualizer.py \
    --port 8080 \
    --workspace ./output-todo-list

# Or monitor multiple tasks
python agents/general_agent/visualizer/visualizer.py \
    --port 8080 \
    --workspaces "output-todo-list,output-file-processor,output-bug-hunter"
```

### Access Dashboard
```
http://localhost:8080/
```

### Real-time Monitoring
```bash
# Terminal 1: Run task
./run_general.sh 01_todo_list --background

# Terminal 2: Start visualizer
python agents/general_agent/visualizer/visualizer.py \
    --port 8080 \
    --workspace ./output-todo-list \
    --live
```

## 🆚 Comparison with Math Agent Visualizer

| Feature | Math Agent | General Agent |
|---------|-----------|---------------|
| **Primary View** | Evolution tree | Task dashboard |
| **Focus** | Solution optimization | Code generation |
| **Key Metric** | Score evolution | Multi-file output |
| **Structure** | Parent-child tree | Linear iterations |
| **Code Display** | Single solution | Multiple files |
| **Diff View** | Parent vs child | Iteration vs iteration |
| **Real-time** | Post-run analysis | Live monitoring |
| **Use Case** | Research problems | Coding tasks |

## 💡 Future Enhancements

1. **Integration with General Agent README**
   - Add visualizer section
   - Update TUTORIAL.md with visualization examples

2. **Video Recording**
   - Record iteration progression as video
   - Time-lapse of code evolution

3. **Collaboration Features**
   - Share task URL
   - Comment on iterations
   - Vote on best solutions

4. **CI/CD Integration**
   - Webhook notifications
   - GitHub Actions integration
   - Slack/Discord alerts

5. **AI Assistant**
   - Ask questions about iterations
   - Get suggestions for task improvements
   - Explain why scores improved/degraded

---

**Ready to implement?** The MVP can be built in ~500 lines of Python + ~800 lines of HTML/CSS/JS.

Let me know if you want me to:
1. Implement the full visualizer
2. Start with just the MVP
3. Modify the design based on your preferences
