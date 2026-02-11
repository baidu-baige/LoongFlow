# 🐛 Example 03: Bug Hunter

**Difficulty**: ⭐⭐⭐ Advanced
**Time**: 15-20 minutes
**Goal**: Build a production-ready bug detection and fixing agent

---

## 🎯 What You'll Learn

This advanced example introduces:

- ✅ **Production-Ready Tools**: Use actual Python analysis tools, not just prompts
- ✅ **Code Analysis**: Static analysis, security scanning, performance profiling
- ✅ **Systematic Debugging**: OWASP Top 10, CWE classifications
- ✅ **Bug Reporting**: Generate professional bug reports with severity levels
- ✅ **Automated Fixing**: Create fixed versions of buggy code

**Prerequisites**: Complete Examples [01](../01_todo_list/) and [02](../02_file_processor/)

---

## 🚀 Quick Start

```bash
# 1. Navigate to LoongFlow root
cd /path/to/LoongFlow

# 2. (Optional) Preview analysis tools manually
cd agents/general_agent/examples/03_bug_hunter
python ../../../.claude/skills/code-analysis/scripts/static_analyzer.py sample_codebase/
python ../../../.claude/skills/code-analysis/scripts/security_scanner.py sample_codebase/

# 3. Run the agent
cd ../../../../
./run_general.sh 03_bug_hunter

# 4. Check the generated bug report
cd output-bug-hunter/task_*/iteration_*/executor/work_dir/
cat BUG_REPORT.md | less
```

---

## 🎨 What Makes This Special?

### Not Just Prompts - Real Tools!

This example includes **3 production Python tools** (~1400 lines total):

1. **static_analyzer.py** (500 lines)
   - AST-based Python analysis
   - Detects 15+ bug patterns
   - Cyclomatic complexity analysis

2. **security_scanner.py** (500 lines)
   - OWASP Top 10 detection
   - CWE vulnerability mapping
   - Severity classification (Critical/High/Medium/Low)

3. **performance_profiler.py** (400 lines)
   - O(n²) loop detection
   - Inefficient algorithm identification
   - Bottleneck analysis

**You can use these tools outside of General Agent too!**

---

## 🧪 Try the Analysis Tools First

Before running the agent, see what the tools can do:

### Static Analysis

```bash
cd agents/general_agent/examples/03_bug_hunter
python ../../../.claude/skills/code-analysis/scripts/static_analyzer.py sample_codebase/
```

**Detects**:
- SQL injection patterns
- Resource leaks (unclosed files)
- Mutable default arguments
- High complexity functions (complexity > 10)
- Missing error handling

**Output Example**:
```
[MEDIUM] sample_codebase/user_manager.py:18
  Issue: Potential resource leak - file not closed

[HIGH] sample_codebase/database_helper.py:35
  Issue: SQL injection vulnerability
  Pattern: f"SELECT * FROM users WHERE id = '{user_id}'"
```

### Security Scanning

```bash
python ../../../.claude/skills/code-analysis/scripts/security_scanner.py sample_codebase/
```

**Detects**:
- CWE-89: SQL Injection
- CWE-798: Hardcoded Credentials
- CWE-327: Weak Cryptography (MD5)
- CWE-95: Code Injection (eval/exec)
- CWE-502: Insecure Deserialization

**Output Example**:
```
[CRITICAL] CWE-89: SQL Injection
  File: database_helper.py:42
  Line: cursor.execute(f"SELECT * FROM users WHERE name = '{name}'")
  Risk: Attacker can execute arbitrary SQL commands

[HIGH] CWE-798: Hardcoded Credentials
  File: api_utils.py:10
  Line: API_KEY = "sk-1234567890"
  Risk: Credentials exposed in source code
```

### Performance Profiling

```bash
python ../../../.claude/skills/code-analysis/scripts/performance_profiler.py sample_codebase/user_manager.py
```

**Detects**:
- Nested loops (O(n²) or worse)
- List membership testing in loops (use set instead)
- String concatenation in loops (use join)
- Repeated function calls in loops

**Output Example**:
```
[PERFORMANCE] Nested loops detected
  Function: search_users (line 58)
  Complexity: O(n²)
  Recommendation: Use dictionary lookup O(1) instead
```

---

## 📦 What the Agent Generates

The Bug Hunter agent produces:

### 1. BUG_REPORT.md
Comprehensive analysis with:
- Executive summary (total issues by severity)
- Detailed issue descriptions with file:line references
- Before/after code examples
- CWE/OWASP classifications
- Remediation recommendations

### 2. fixed_code/
Complete fixed versions of all files:
- `user_manager.py` (fixed)
- `database_helper.py` (fixed)
- `api_utils.py` (fixed)

Each fix includes comments explaining what was changed.

### 3. CHANGES.md
Summary of all modifications:
- Security fixes applied
- Bugs resolved
- Performance improvements
- Code quality enhancements

---

## 🧩 Sample Codebase (Intentionally Buggy)

Located at [sample_codebase/](sample_codebase/):

### user_manager.py
User management system with bugs:
- Resource leaks (unclosed files)
- Mutable default arguments
- Missing null checks
- Off-by-one errors

### database_helper.py
Database operations with:
- SQL injection vulnerabilities (8+ instances)
- Unclosed connections
- Transaction management issues
- No parameterized queries

### api_utils.py
API utilities with:
- Hardcoded credentials
- Plain text password storage
- Weak hashing (MD5)
- Missing input validation

**Total**: 38 intentional bugs across 3 files

---

## 🎯 What the Agent Does

### Phase 1: Detection

1. Runs static analyzer → Finds code quality issues
2. Runs security scanner → Identifies vulnerabilities
3. Runs performance profiler → Detects inefficiencies
4. Manual code review → Catches issues tools miss
5. Combines all findings → Comprehensive bug list

### Phase 2: Reporting

Creates `BUG_REPORT.md` with:
- Issues categorized by severity
- CWE/OWASP mappings
- Clear explanations
- Fix recommendations

### Phase 3: Fixing

1. Fixes critical issues first (security)
2. Then high priority (crashes, data loss)
3. Then medium (performance, resource leaks)
4. Creates `fixed_code/` directory
5. Documents all changes in `CHANGES.md`

---

## 🧪 Test the Generated Output

```bash
cd output-bug-hunter/task_*/iteration_*/executor/work_dir/

# 1. Read the bug report
cat BUG_REPORT.md | less

# Should have ~30-40 issues found, categorized by severity

# 2. Check the changes summary
cat CHANGES.md

# Should list all fixes made

# 3. Compare before/after
diff sample_codebase/database_helper.py fixed_code/database_helper.py

# Should see SQL injection fixes, parameterized queries, etc.

# 4. Verify fixed code works
cd fixed_code/
python -m py_compile *.py  # Should compile without errors
```

---

## ⚙️ Configuration Highlights

Key features in [task_config.yaml](task_config.yaml):

```yaml
planners:
  general_planner:
    skills: ["code-analysis"]      # 🎯 Load analysis tools
    max_turns: 50                  # More turns for thorough analysis

evolve:
  task: |
    You are a Bug Hunter Agent with access to powerful analysis tools!

    Phase 1: Run these tools to detect issues:
      python .claude/skills/code-analysis/scripts/static_analyzer.py sample_codebase/
      python .claude/skills/code-analysis/scripts/security_scanner.py sample_codebase/
      python .claude/skills/code-analysis/scripts/performance_profiler.py sample_codebase/*.py

    Phase 2: Generate BUG_REPORT.md with all findings

    Phase 3: Create fixed_code/ directory with corrected files

  max_iterations: 30
  target_score: 0.90               # High quality threshold

  evaluator:
    timeout: 900                   # 15 minutes for evaluation
```

**Important**: The task explicitly tells the agent about available tools!

---

## 📚 Learning Resources Included

The code-analysis skill includes comprehensive reference documentation:

### 1. OWASP Top 10 for Python
[.claude/skills/code-analysis/references/owasp_top10_python.md]

600+ lines covering:
- A01: Broken Access Control
- A02: Cryptographic Failures
- A03: Injection (SQL, Command, XSS)
- A04-A10: All OWASP categories

Each with ❌ Bad and ✅ Good Python examples.

### 2. CWE Quick Reference
[.claude/skills/code-analysis/references/cwe_quick_reference.md]

300+ lines covering essential CWEs:
- CWE-20: Input Validation
- CWE-78: Command Injection
- CWE-89: SQL Injection
- CWE-79: XSS
- And 10+ more

### 3. Tools Usage Guide
[.claude/skills/code-analysis/references/TOOLS_USAGE.md]

400+ lines explaining:
- How to use each tool
- CI/CD integration examples
- Customization guide
- Output format specs

**Read these to understand security patterns!**

---

## 🎓 Expected Results

### Iteration 1 (Score: 0.5-0.6)
- Tools run successfully
- Basic bug report generated
- Some bugs identified

### Iteration 2 (Score: 0.7-0.8)
- Comprehensive bug report
- Most bugs found
- Fixed code created (partial)

### Iteration 3 (Score: 0.85-0.95) ✅
- All 38 bugs identified
- Professional report with CWE mappings
- Complete fixed_code/ directory
- Detailed CHANGES.md

---

## 🛠️ Common Issues & Solutions

### Issue: "Tools not found"

```bash
# Check tool exists and is executable
ls -la .claude/skills/code-analysis/scripts/*.py

# Make executable if needed
chmod +x .claude/skills/code-analysis/scripts/*.py

# Test manually
python .claude/skills/code-analysis/scripts/static_analyzer.py --help
```

### Issue: "No bugs detected"

```bash
# Verify sample codebase exists
ls -la agents/general_agent/examples/03_bug_hunter/sample_codebase/

# Check if agent actually ran the tools
grep -r "static_analyzer" output-bug-hunter/task_*/iteration_*/planner/*.log

# Agent might have missed tool instructions - check task prompt
```

### Issue: "Score always low"

- Bug detection might be incomplete
- Check evaluator logs to see what's missing
- Increase max_turns to give agent more time
- Verify all 3 analysis tools ran successfully

---

## 🏆 Challenge: Find All 38 Bugs!

The sample codebase has exactly **38 intentional bugs**. Can your agent find them all?

**Bug Categories**:
- Security: 12 bugs (SQL injection, weak crypto, hardcoded secrets)
- Logic: 8 bugs (null checks, off-by-one, wrong conditions)
- Resources: 6 bugs (unclosed files, connections, missing commits)
- Performance: 5 bugs (O(n²) loops, inefficient algorithms)
- Quality: 7 bugs (mutable defaults, bare except, global state)

**Perfect Score**: All 38 bugs found, correctly categorized, and fixed!

---

## 🎉 Next Steps

Excellent work building a production-ready analysis agent! Now try:

1. **Analyze your own code**: Point the agent at your projects
   ```yaml
   task: |
     Analyze the codebase at /path/to/my/project
   ```

2. **Customize the tools**: Add domain-specific checks
   - Edit `static_analyzer.py` to add new patterns
   - Add your company's security policies to scanner

3. **Integrate into CI/CD**: See TOOLS_USAGE.md for GitHub Actions examples

4. **Try Example 04**: [04_circle_packing](../04_circle_packing/) - Advanced optimization problems

---

## 📚 Key Takeaways

- ✅ **Production tools** make skills powerful, not just prompts
- ✅ **Systematic analysis** beats ad-hoc bug hunting
- ✅ **OWASP/CWE classifications** make findings professional
- ✅ **Automated fixing** saves hours of manual work
- ✅ **Reusable tools** can be used outside the agent

**Ready for expert-level challenges?** Head to [Example 04: Circle Packing](../04_circle_packing/) for advanced optimization! 🚀
