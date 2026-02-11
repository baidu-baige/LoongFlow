---
name: "code-analysis"
description: "Code review and debugging assistant. Identifies bugs, performance issues, security vulnerabilities, and suggests optimizations."
---

# Code Analysis & Bug Hunting Skill

This skill provides systematic approaches for analyzing code, identifying issues, and proposing fixes.

## Purpose

Use this skill when your task involves:
- Finding bugs in existing code
- Code quality review
- Performance optimization suggestions
- Security vulnerability detection
- Refactoring recommendations
- Best practices enforcement

## Key Capabilities

### 1. Bug Detection
- Logic errors (off-by-one, null checks, etc.)
- Type mismatches and conversion issues
- Resource leaks (file handles, connections)
- Concurrency issues (race conditions, deadlocks)
- Exception handling gaps
- Edge case failures

### 2. Code Quality Issues
- Code duplication
- Overly complex functions
- Poor naming conventions
- Missing documentation
- Inconsistent formatting
- Dead code

### 3. Performance Problems
- Inefficient algorithms (O(n²) where O(n) possible)
- Unnecessary loops or operations
- Memory leaks
- Excessive I/O operations
- Missing caching opportunities
- Database query inefficiencies

### 4. Security Vulnerabilities
- SQL injection risks
- XSS vulnerabilities
- Insecure authentication
- Exposed secrets/credentials
- Insufficient input validation
- Path traversal risks

## Bug Hunting Workflow

### Phase 1: Code Understanding
```
1. Read all files to understand codebase structure
2. Identify entry points and main workflows
3. Map dependencies between components
4. Note external dependencies and APIs used
```

### Phase 2: Issue Detection
```
1. Static analysis (read code carefully)
2. Look for common bug patterns
3. Check error handling coverage
4. Validate input/output handling
5. Review edge cases
```

### Phase 3: Issue Documentation
```
For each issue found, document:
- Location: file:line_number
- Severity: Critical / High / Medium / Low
- Category: Bug / Performance / Security / Quality
- Description: What's wrong
- Impact: What could go wrong
- Recommendation: How to fix it
- Code example: Before and after
```

### Phase 4: Fix Implementation
```
1. Prioritize by severity
2. Implement fixes one at a time
3. Preserve existing functionality
4. Add comments explaining changes
5. Ensure backward compatibility (if needed)
```

## Bug Report Template

```markdown
## Bug Report

### Issue #N: [Brief Title]

**Location**: `file_path:line_number`
**Severity**: Critical | High | Medium | Low
**Category**: Bug | Performance | Security | Quality

**Description**:
[Clear explanation of what's wrong]

**Current Code**:
```python
# Problematic code here
```

**Issue**:
[Why this is a problem]

**Impact**:
- [What could go wrong]
- [Example failure scenario]

**Recommended Fix**:
```python
# Fixed code here
```

**Explanation**:
[Why this fix works]
```

## Common Bug Patterns

### 1. Null/None Check Missing
```python
# ❌ Bug
def process_user(user):
    return user.name.upper()  # Crashes if user is None

# ✅ Fix
def process_user(user):
    if user is None:
        return None
    return user.name.upper()
```

### 2. Off-by-One Error
```python
# ❌ Bug
for i in range(len(items) - 1):  # Misses last item!
    process(items[i])

# ✅ Fix
for i in range(len(items)):
    process(items[i])
# Or better:
for item in items:
    process(item)
```

### 3. Resource Leak
```python
# ❌ Bug
def read_file(path):
    f = open(path)
    data = f.read()
    return data  # File never closed!

# ✅ Fix
def read_file(path):
    with open(path) as f:
        return f.read()
```

### 4. Mutable Default Argument
```python
# ❌ Bug
def append_to(element, target=[]):
    target.append(element)
    return target  # Shares same list across calls!

# ✅ Fix
def append_to(element, target=None):
    if target is None:
        target = []
    target.append(element)
    return target
```

### 5. Exception Swallowing
```python
# ❌ Bug
try:
    result = risky_operation()
except:
    pass  # Silent failure, no one knows what went wrong

# ✅ Fix
try:
    result = risky_operation()
except ValueError as e:
    logging.error(f"Failed to process: {e}")
    raise
```

### 6. SQL Injection
```python
# ❌ Bug
query = f"SELECT * FROM users WHERE name = '{user_input}'"
cursor.execute(query)  # Vulnerable to injection!

# ✅ Fix
query = "SELECT * FROM users WHERE name = ?"
cursor.execute(query, (user_input,))
```

### 7. Inefficient Loop
```python
# ❌ Performance Issue
result = []
for item in large_list:
    if item not in result:  # O(n²) complexity!
        result.append(item)

# ✅ Fix
result = list(set(large_list))  # O(n)
# Or if order matters:
seen = set()
result = []
for item in large_list:
    if item not in seen:
        seen.add(item)
        result.append(item)
```

## Analysis Tools Available

When analyzing code, you have access to:
- `Read` - Read source files
- `Grep` - Search for patterns across codebase
- `Glob` - Find files by name/extension
- `Bash` - Run linters (pylint, flake8, mypy)

## Best Practices for Bug Fixing

### 1. Minimal Changes
- Fix one bug at a time
- Don't refactor while fixing bugs
- Preserve existing behavior (unless that's the bug)

### 2. Clear Documentation
```python
# Before: Bug - doesn't handle negative numbers
# After: Fixed - added validation for negative inputs
def calculate_square_root(num):
    if num < 0:
        raise ValueError("Cannot calculate square root of negative number")
    return num ** 0.5
```

### 3. Add Defensive Checks
```python
def divide(a, b):
    """Divide a by b with safety checks"""
    if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
        raise TypeError("Both arguments must be numbers")
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b
```

### 4. Preserve Compatibility
```python
# When fixing a public API, maintain backward compatibility
def old_function(param1):
    """Deprecated: Use new_function instead"""
    warnings.warn("old_function is deprecated", DeprecationWarning)
    return new_function(param1)

def new_function(param1, param2=None):
    """New implementation with additional features"""
    # ... improved logic
```

## Output Format

### Bug Analysis Report
```markdown
# Code Analysis Report

## Summary
- Total files analyzed: X
- Issues found: Y
- Critical: N
- High: N
- Medium: N
- Low: N

## Critical Issues
[List critical bugs that need immediate attention]

## High Priority Issues
[List high priority bugs]

## Medium Priority Issues
[List medium priority issues]

## Low Priority Issues
[List code quality suggestions]

## Recommended Actions
1. [Priority 1 fix]
2. [Priority 2 fix]
...
```

### Fixed Code Structure
```
fixed_code/
├── file1.py          # Fixed version with comments
├── file2.py          # Fixed version with comments
└── CHANGES.md        # Summary of all changes made
```

## Interactive Workflow

When working with users:

1. **Present Findings**: Show bug report with severity
2. **Wait for Approval**: Ask user which bugs to fix
3. **Implement Fixes**: Apply approved fixes only
4. **Document Changes**: Create CHANGES.md explaining what was fixed

## Example Task Breakdown

**Task**: "Find and fix bugs in this codebase"

**Step 1**: Analyze
- Read all Python files
- Identify potential issues
- Categorize by severity

**Step 2**: Report
- Generate bug report
- Prioritize issues
- Present to user

**Step 3**: Fix (after approval)
- Implement fixes for approved issues
- Test that fixes work
- Document changes

**Step 4**: Verify
- Review fixed code
- Ensure no new bugs introduced
- Provide summary

## Testing Recommendations

After fixing bugs, suggest:
```python
# Add tests for the bug fix
def test_divide_by_zero():
    """Ensure divide by zero raises ValueError"""
    with pytest.raises(ValueError):
        divide(10, 0)

def test_divide_negative_numbers():
    """Ensure negative numbers work correctly"""
    assert divide(-10, 2) == -5.0
```

## Security Checklist

When analyzing code for security:
- [ ] All user inputs are validated
- [ ] No SQL queries use string concatenation
- [ ] Passwords/secrets are not hardcoded
- [ ] File paths are validated (no traversal)
- [ ] Authentication is required for sensitive operations
- [ ] Error messages don't leak sensitive info
- [ ] Dependencies are up to date (no known vulnerabilities)

## Performance Checklist

When analyzing for performance:
- [ ] No nested loops that could be optimized
- [ ] Database queries use proper indexes
- [ ] Caching is used where appropriate
- [ ] Large files are read in chunks
- [ ] Expensive operations are memoized
- [ ] Unnecessary computations are avoided

## Quality Checklist

When analyzing code quality:
- [ ] Functions are < 50 lines
- [ ] Clear, descriptive variable names
- [ ] Comments explain "why", not "what"
- [ ] No code duplication
- [ ] Error messages are helpful
- [ ] Type hints are present
- [ ] Docstrings document public APIs
- [ ] Docstrings document public APIs

---

## 🛠️ Available Analysis Tools

This skill comes with production-ready Python tools that you can use directly via Bash:

### 1. Static Analyzer
Comprehensive code analysis for bugs, security, and quality.

```bash
# Analyze entire codebase
python .claude/skills/code-analysis/scripts/static_analyzer.py <directory>

# Generate JSON report
python .claude/skills/code-analysis/scripts/static_analyzer.py <directory> \
  --format json --output analysis.json
```

**Detects**:
- Security vulnerabilities (SQL injection, eval/exec, etc.)
- Logic bugs (null checks, off-by-one errors)
- Resource leaks (unclosed files)
- Code quality issues (complexity, mutable defaults)

### 2. Performance Profiler
Identifies performance bottlenecks and algorithmic inefficiencies.

```bash
# Analyze performance
python .claude/skills/code-analysis/scripts/performance_profiler.py <file.py>

# With custom threshold
python .claude/skills/code-analysis/scripts/performance_profiler.py <file.py> --threshold 3
```

**Detects**:
- Nested loops (O(n²) complexity)
- List membership tests in loops (use sets)
- Inefficient string operations
- Repeated global lookups

### 3. Security Scanner
Specialized OWASP Top 10 and CWE vulnerability scanner.

```bash
# Security scan
python .claude/skills/code-analysis/scripts/security_scanner.py <directory>

# Exit code 1 if critical vulnerabilities found
python .claude/skills/code-analysis/scripts/security_scanner.py <directory> \
  --output security_report.txt
```

**Detects**:
- SQL injection (CWE-89)
- Command injection (CWE-78)
- Hardcoded credentials (CWE-798)
- Weak cryptography (CWE-327)
- Insecure deserialization (CWE-502)

---

## 📚 Reference Documentation

Comprehensive security guides available in the skill package:

### OWASP Top 10 for Python
[references/owasp_top10_python.md](references/owasp_top10_python.md)

- Practical examples for each OWASP Top 10 category
- Python-specific security patterns
- Before/after code examples
- Remediation strategies

### CWE Quick Reference
[references/cwe_quick_reference.md](references/cwe_quick_reference.md)

- Common Weakness Enumeration essentials
- Python examples for each CWE
- Severity levels and priority guidance
- Testing strategies

### Tools Usage Guide
[references/TOOLS_USAGE.md](references/TOOLS_USAGE.md)

- Complete documentation for all analysis tools
- CI/CD integration examples
- Pre-commit hook samples
- Customization guide

---

## 🎯 Recommended Workflow for Bug Hunting

### Phase 1: Automated Scanning (5-10 minutes)

```bash
# Run all three tools
python .claude/skills/code-analysis/scripts/static_analyzer.py . --output static.txt
python .claude/skills/code-analysis/scripts/security_scanner.py . --output security.txt

# For large files, also profile performance
find . -name "*.py" -size +100k -exec \
  python .claude/skills/code-analysis/scripts/performance_profiler.py {} \; \
  > perf.txt
```

### Phase 2: Manual Analysis (10-20 minutes)

1. **Read tool outputs** to understand detected issues
2. **Prioritize by severity**: Critical > High > Medium > Low
3. **Verify findings**: Some may be false positives
4. **Group related issues**: E.g., all SQL injections together

### Phase 3: Report Generation (15-30 minutes)

Create comprehensive bug report:
- Executive summary with counts
- Detailed findings with file:line references
- Code snippets (before/after)
- Remediation recommendations
- References to OWASP/CWE

### Phase 4: Fix Implementation (Variable time)

Implement fixes:
- One category at a time (e.g., all SQL injections)
- Add comments explaining each fix
- Don't introduce new bugs
- Preserve original functionality

### Phase 5: Verification (10-15 minutes)

Re-run tools to verify:
```bash
# Should show fewer issues
python .claude/skills/code-analysis/scripts/security_scanner.py .
```

---

## 💡 Pro Tips

### Combining Tools with Manual Analysis

```python
# Example workflow in agent code
import subprocess
import json

# Run static analyzer
result = subprocess.run([
    'python', '.claude/skills/code-analysis/scripts/static_analyzer.py',
    target_dir, '--format', 'json'
], capture_output=True, text=True)

issues = json.loads(result.stdout)

# Filter critical issues
critical = [i for i in issues if i['severity'] == 'critical']

# Generate detailed report for each
for issue in critical:
    # Read the problematic code
    with open(issue['file']) as f:
        lines = f.readlines()
        context = lines[issue['line']-3:issue['line']+2]

    # Add to bug report with context
    report.add_issue(issue, context)
```

### Custom Analysis Patterns

You can extend the tools by adding custom checkers. See [references/TOOLS_USAGE.md](references/TOOLS_USAGE.md) for examples.

### Integration with CI/CD

```yaml
# GitHub Actions example
- name: Security Scan
  run: |
    python .claude/skills/code-analysis/scripts/security_scanner.py . \
      --format json --output security.json

- name: Check Results
  run: |
    CRITICAL=$(jq '[.[] | select(.severity=="critical")] | length' security.json)
    if [ "$CRITICAL" -gt 0 ]; then
      echo "Found $CRITICAL critical issues"
      exit 1
    fi
```

---

## 🎓 Learning Resources

The reference documents contain extensive learning material:

1. **Start with OWASP Top 10** - Learn the most common security risks
2. **Study CWE examples** - Understand specific vulnerability types
3. **Practice with tools** - Run them on sample code
4. **Read tool source code** - See how detection works

---

## 🚀 Real-World Usage

This skill package is production-ready and includes tools that:

✅ Have been tested on real codebases
✅ Produce actionable, detailed reports
✅ Support both human and machine-readable output
✅ Include comprehensive error handling
✅ Work with CI/CD pipelines
✅ Scale to large codebases

The tools themselves follow security best practices and can be used as reference implementations.

---

## 📖 Additional Resources

- OWASP Top 10: https://owasp.org/www-project-top-ten/
- CWE Top 25: https://cwe.mitre.org/top25/
- Python Security: https://python.readthedocs.io/en/latest/library/security_warnings.html
- Bandit (inspiration): https://github.com/PyCQA/bandit
- OWASP Cheat Sheets: https://cheatsheetseries.owasp.org/

---

**Remember**: Security is a process, not a product. Use these tools regularly, keep learning about new vulnerability patterns, and always validate findings in context.
