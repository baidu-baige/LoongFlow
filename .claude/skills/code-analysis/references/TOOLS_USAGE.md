# Code Analysis Tools Usage Guide

This directory contains powerful Python scripts for automated code analysis. These tools can be used directly via Bash or imported by the agent.

## Available Tools

### 1. Static Analyzer (`static_analyzer.py`)

Comprehensive static code analysis tool that detects bugs, security issues, and code quality problems.

**Features**:
- AST-based Python code analysis
- Security vulnerability detection
- Performance issue identification
- Code quality checks
- Configurable severity levels

**Usage**:
```bash
# Analyze a directory
python scripts/static_analyzer.py /path/to/codebase

# Save report to file
python scripts/static_analyzer.py /path/to/code --output report.txt

# JSON format for programmatic processing
python scripts/static_analyzer.py /path/to/code --format json --output report.json
```

**Example Output**:
```
================================================================================
CODE ANALYSIS REPORT
================================================================================

SUMMARY
--------------------------------------------------------------------------------
Total issues: 15
Critical: 2
High: 5
Medium: 6
Low: 2

By Category:
  security: 3
  bug: 7
  performance: 3
  quality: 2

================================================================================
CRITICAL ISSUES (2)
================================================================================

[SQL_INJECTION] Possible SQL injection via string formatting
  File: database.py:45:12
  Category: security
  Suggestion: Use parameterized queries with placeholders
...
```

**Detected Issues**:
- Dangerous functions (eval, exec, pickle.loads)
- SQL injection patterns
- Resource leaks (unclosed files)
- Mutable default arguments
- High cyclomatic complexity
- Missing docstrings
- Bare except clauses
- Identity comparisons with literals

---

### 2. Performance Profiler (`performance_profiler.py`)

Identifies performance bottlenecks and algorithmic inefficiencies.

**Features**:
- Nested loop detection (O(n²) complexity)
- List membership testing in loops
- Inefficient string operations
- Repeated function calls in loops
- Algorithmic anti-patterns

**Usage**:
```bash
# Analyze a Python file
python scripts/performance_profiler.py script.py

# Set complexity threshold
python scripts/performance_profiler.py script.py --threshold 3

# Save report
python scripts/performance_profiler.py script.py --output perf_report.txt
```

**Example Output**:
```
================================================================================
PERFORMANCE ANALYSIS REPORT
================================================================================
File: data_processor.py

SUMMARY
--------------------------------------------------------------------------------
Total performance issues: 8
Nested loops (O(n²) or worse): 3
Inefficient list operations: 3
String operation issues: 2

DETAILED ISSUES
================================================================================

Issue #1: NESTED_LOOPS
--------------------------------------------------------------------------------
Line: 45
Description: Nested loop detected (depth: 2)
Current Complexity: O(n^2)
Optimized Complexity: O(n) or O(n log n)

Code:
  for user in users:
      for item in items:
          if user.id == item.user_id:

💡 Suggestion: Use dict for lookups, or list comprehension
...
```

**Performance Patterns Detected**:
- Nested loops
- List membership tests in loops (use sets)
- Sorting inside loops
- Unnecessary `.keys()` on dicts
- String concatenation in loops
- Repeated global lookups

---

### 3. Security Scanner (`security_scanner.py`)

Specialized security-focused scanner that detects OWASP Top 10 and CWE vulnerabilities.

**Features**:
- OWASP Top 10 vulnerability detection
- CWE (Common Weakness Enumeration) mapping
- Hardcoded credentials detection
- Weak cryptography identification
- SSRF and injection vulnerability detection
- Compliance with security standards

**Usage**:
```bash
# Scan a directory
python scripts/security_scanner.py /path/to/codebase

# Save report
python scripts/security_scanner.py /path/to/code --output security_report.txt

# JSON output for CI/CD integration
python scripts/security_scanner.py /path/to/code --format json --output security.json

# Exit code 1 if critical vulnerabilities found
echo $?  # Check exit code in CI/CD
```

**Example Output**:
```
==================================================================================
SECURITY SCAN REPORT
==================================================================================

EXECUTIVE SUMMARY
----------------------------------------------------------------------------------
Total vulnerabilities found: 12
  🔴 Critical: 4
  🟠 High: 3
  🟡 Medium: 3
  🔵 Low: 2

⚠️  CRITICAL vulnerabilities found - immediate action required!

==================================================================================
🔴 CRITICAL SEVERITY (4 findings)
==================================================================================

[CWE-89] SQL Injection
  File: database_helper.py:20
  Description: Possible SQL injection via string concatenation
  Evidence: sql = f"SELECT * FROM users WHERE name = '{params}'"
  🔧 Remediation: Use parameterized queries with placeholders (?, %s)
  📚 References:
     https://cwe.mitre.org/data/definitions/89.html
     https://cheatsheetseries.owasp.org/cheatsheets/SQL_Injection_Prevention_Cheat_Sheet.html

[CWE-798] Hardcoded Password
  File: config.py:15
  Description: Detected: Hardcoded Password
  Evidence: password = "admin123"
  🔧 Remediation: Use environment variables or secure config management
  📚 References:
     https://cwe.mitre.org/data/definitions/798.html
...
```

**Detected Vulnerabilities**:
- SQL Injection (CWE-89)
- Command Injection (CWE-78)
- Code Injection (CWE-95)
- Hardcoded credentials (CWE-798)
- Weak cryptography (CWE-327)
- Insecure deserialization (CWE-502)
- Weak random numbers (CWE-330)
- Disabled SSL verification (CWE-295)
- Debug mode in production (CWE-489)

---

## Integration with Agent

The agent can use these tools via Bash:

```python
# Example agent usage
from loongflow.agentsdk.tools import BashTool

# Run static analysis
result = await bash_tool.execute(
    "python .claude/skills/code-analysis/scripts/static_analyzer.py sample_code/"
)

# Parse results and generate report
issues = parse_analysis_output(result.output)
report = generate_bug_report(issues)
```

---

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Code Security Scan

on: [push, pull_request]

jobs:
  security:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2

      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.12'

      - name: Run Security Scanner
        run: |
          python .claude/skills/code-analysis/scripts/security_scanner.py . \
            --output security_report.txt \
            --format json

      - name: Upload Report
        uses: actions/upload-artifact@v2
        if: always()
        with:
          name: security-report
          path: security_report.txt
```

### Pre-commit Hook Example

```bash
#!/bin/bash
# .git/hooks/pre-commit

echo "Running security scan..."
python .claude/skills/code-analysis/scripts/security_scanner.py . --format json > /tmp/security.json

CRITICAL_COUNT=$(jq '[.[] | select(.severity=="critical")] | length' /tmp/security.json)

if [ "$CRITICAL_COUNT" -gt 0 ]; then
    echo "❌ Found $CRITICAL_COUNT critical security issues"
    echo "Run: python .claude/skills/code-analysis/scripts/security_scanner.py . --output report.txt"
    exit 1
fi

echo "✓ No critical security issues found"
```

---

## Output Formats

### Text Format
Human-readable reports with:
- Executive summary
- Issues grouped by severity
- Detailed descriptions
- Code snippets
- Remediation advice

### JSON Format
Machine-readable for automation:
```json
[
  {
    "file": "user_manager.py",
    "line": 35,
    "severity": "critical",
    "category": "security",
    "code": "SQL_INJECTION",
    "message": "Possible SQL injection",
    "suggestion": "Use parameterized queries"
  }
]
```

---

## Best Practices

### 1. Run Tools Regularly
```bash
# Weekly security scan
0 0 * * 0 python security_scanner.py /app > weekly_security.txt

# Daily static analysis
0 2 * * * python static_analyzer.py /app > daily_analysis.txt
```

### 2. Prioritize Issues
1. **Critical**: Fix immediately (security vulnerabilities)
2. **High**: Fix within days (crashes, data loss)
3. **Medium**: Fix within weeks (performance, quality)
4. **Low**: Fix when convenient (style, documentation)

### 3. Baseline and Track
```bash
# Create baseline
python static_analyzer.py . --format json > baseline.json

# Compare with baseline
python static_analyzer.py . --format json > current.json
diff baseline.json current.json
```

### 4. Combine Tools
```bash
# Comprehensive analysis
python static_analyzer.py . --output static.txt
python performance_profiler.py main.py --output perf.txt
python security_scanner.py . --output security.txt

# Aggregate results
cat static.txt perf.txt security.txt > full_analysis.txt
```

---

## Customization

### Adding Custom Checks

```python
# Example: Add custom checker to static_analyzer.py

class CustomChecker:
    @staticmethod
    def check_file(filepath: str, content: str) -> List[Issue]:
        issues = []
        lines = content.split('\n')

        for i, line in enumerate(lines, 1):
            if 'TODO' in line:
                issues.append(Issue(
                    file=filepath,
                    line=i,
                    column=0,
                    severity="low",
                    category="quality",
                    code="TODO_COMMENT",
                    message="TODO comment found",
                    suggestion="Complete or remove TODO"
                ))

        return issues

# Then add to analyzer:
self.issues.extend(CustomChecker.check_file(str(filepath), content))
```

---

## Troubleshooting

### Tool Not Found
```bash
# Ensure you're in the right directory
cd /path/to/LoongFlow
ls .claude/skills/code-analysis/scripts/

# Make scripts executable
chmod +x .claude/skills/code-analysis/scripts/*.py
```

### Permission Errors
```bash
# Run with proper permissions
python3 scripts/static_analyzer.py .
```

### Large Codebases
```bash
# Analyze specific subdirectories
python scripts/static_analyzer.py src/
python scripts/static_analyzer.py tests/

# Or parallelize
find . -name "*.py" -print0 | xargs -0 -P 4 -I {} python scripts/performance_profiler.py {}
```

---

## Performance Tips

- Tools are I/O bound, so SSDs help
- Static analyzer is single-threaded; parallelize for large repos
- JSON output is faster than text formatting
- Cache results and run incrementally on changed files only

---

## Support

For issues or questions:
1. Check error messages carefully
2. Verify Python version (3.12+ required)
3. Ensure file paths are correct
4. Review the skill's SKILL.md for context

These tools are production-ready and used to analyze the LoongFlow codebase itself!
