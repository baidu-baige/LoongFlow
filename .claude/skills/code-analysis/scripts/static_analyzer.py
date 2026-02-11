#!/usr/bin/env python3
"""
Static Code Analyzer
Performs comprehensive static analysis on Python codebases
Usage: python static_analyzer.py <directory> [--output OUTPUT] [--format json|text]
"""

import ast
import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Set
from dataclasses import dataclass, asdict
from collections import defaultdict


@dataclass
class Issue:
    """Represents a code issue"""
    file: str
    line: int
    column: int
    severity: str  # critical, high, medium, low
    category: str  # security, bug, performance, quality
    code: str
    message: str
    suggestion: str = ""


class PythonAnalyzer(ast.NodeVisitor):
    """AST-based Python code analyzer"""

    def __init__(self, filepath: str):
        self.filepath = filepath
        self.issues: List[Issue] = []
        self.current_function = None
        self.imported_modules: Set[str] = set()

    def analyze(self, source_code: str) -> List[Issue]:
        """Analyze Python source code"""
        try:
            tree = ast.parse(source_code)
            self.visit(tree)
        except SyntaxError as e:
            self.issues.append(Issue(
                file=self.filepath,
                line=e.lineno or 0,
                column=e.offset or 0,
                severity="critical",
                category="bug",
                code="SYNTAX_ERROR",
                message=f"Syntax error: {e.msg}",
                suggestion="Fix the syntax error before proceeding"
            ))
        return self.issues

    def visit_Import(self, node):
        """Check import statements"""
        for alias in node.names:
            self.imported_modules.add(alias.name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        """Check from-import statements"""
        if node.module:
            self.imported_modules.add(node.module)
        self.generic_visit(node)

    def visit_FunctionDef(self, node):
        """Analyze function definitions"""
        prev_function = self.current_function
        self.current_function = node.name

        # Check function complexity
        complexity = self._calculate_complexity(node)
        if complexity > 10:
            self.issues.append(Issue(
                file=self.filepath,
                line=node.lineno,
                column=node.col_offset,
                severity="medium",
                category="quality",
                code="HIGH_COMPLEXITY",
                message=f"Function '{node.name}' has high complexity ({complexity})",
                suggestion="Consider breaking down into smaller functions"
            ))

        # Check function length
        func_length = node.end_lineno - node.lineno if node.end_lineno else 0
        if func_length > 50:
            self.issues.append(Issue(
                file=self.filepath,
                line=node.lineno,
                column=node.col_offset,
                severity="low",
                category="quality",
                code="LONG_FUNCTION",
                message=f"Function '{node.name}' is too long ({func_length} lines)",
                suggestion="Keep functions under 50 lines for better maintainability"
            ))

        # Check for missing docstring
        if not ast.get_docstring(node):
            self.issues.append(Issue(
                file=self.filepath,
                line=node.lineno,
                column=node.col_offset,
                severity="low",
                category="quality",
                code="MISSING_DOCSTRING",
                message=f"Function '{node.name}' lacks a docstring",
                suggestion="Add a docstring explaining the function's purpose"
            ))

        # Check for mutable default arguments
        for default in node.args.defaults:
            if isinstance(default, (ast.List, ast.Dict, ast.Set)):
                self.issues.append(Issue(
                    file=self.filepath,
                    line=node.lineno,
                    column=node.col_offset,
                    severity="high",
                    category="bug",
                    code="MUTABLE_DEFAULT",
                    message=f"Function '{node.name}' has mutable default argument",
                    suggestion="Use None as default and initialize inside function"
                ))

        self.generic_visit(node)
        self.current_function = prev_function

    def visit_Call(self, node):
        """Analyze function calls"""
        # Check for dangerous functions
        if isinstance(node.func, ast.Name):
            func_name = node.func.id

            # Dangerous eval/exec
            if func_name in ('eval', 'exec'):
                self.issues.append(Issue(
                    file=self.filepath,
                    line=node.lineno,
                    column=node.col_offset,
                    severity="critical",
                    category="security",
                    code="DANGEROUS_EVAL",
                    message=f"Use of dangerous function '{func_name}'",
                    suggestion="Avoid eval/exec; use safer alternatives"
                ))

            # Open without context manager
            if func_name == 'open':
                # Check if inside 'with' statement
                if not self._is_in_with_context(node):
                    self.issues.append(Issue(
                        file=self.filepath,
                        line=node.lineno,
                        column=node.col_offset,
                        severity="high",
                        category="bug",
                        code="RESOURCE_LEAK",
                        message="File opened without context manager",
                        suggestion="Use 'with open(...) as f:' to ensure file is closed"
                    ))

        # Check for SQL injection patterns
        if isinstance(node.func, ast.Attribute):
            if node.func.attr == 'execute':
                for arg in node.args:
                    if isinstance(arg, ast.JoinedStr) or isinstance(arg, ast.BinOp):
                        self.issues.append(Issue(
                            file=self.filepath,
                            line=node.lineno,
                            column=node.col_offset,
                            severity="critical",
                            category="security",
                            code="SQL_INJECTION",
                            message="Possible SQL injection: query uses string formatting",
                            suggestion="Use parameterized queries with placeholders"
                        ))

        self.generic_visit(node)

    def visit_Try(self, node):
        """Analyze try-except blocks"""
        for handler in node.handlers:
            # Check for bare except
            if handler.type is None:
                self.issues.append(Issue(
                    file=self.filepath,
                    line=handler.lineno,
                    column=handler.col_offset,
                    severity="medium",
                    category="quality",
                    code="BARE_EXCEPT",
                    message="Bare 'except:' catches all exceptions",
                    suggestion="Catch specific exception types"
                ))

            # Check for pass in except
            if len(handler.body) == 1 and isinstance(handler.body[0], ast.Pass):
                self.issues.append(Issue(
                    file=self.filepath,
                    line=handler.lineno,
                    column=handler.col_offset,
                    severity="high",
                    category="bug",
                    code="SILENT_EXCEPTION",
                    message="Exception silently ignored with 'pass'",
                    suggestion="Log the exception or re-raise if needed"
                ))

        self.generic_visit(node)

    def visit_For(self, node):
        """Analyze for loops"""
        # Check for list modification during iteration
        if isinstance(node.iter, ast.Name):
            for stmt in ast.walk(node):
                if isinstance(stmt, ast.Call):
                    if isinstance(stmt.func, ast.Attribute):
                        if (stmt.func.attr in ('remove', 'pop', 'append') and
                            isinstance(stmt.func.value, ast.Name) and
                            stmt.func.value.id == node.iter.id):
                            self.issues.append(Issue(
                                file=self.filepath,
                                line=node.lineno,
                                column=node.col_offset,
                                severity="critical",
                                category="bug",
                                code="MODIFY_DURING_ITERATION",
                                message="Modifying list while iterating over it",
                                suggestion="Iterate over a copy or use list comprehension"
                            ))

        self.generic_visit(node)

    def visit_Compare(self, node):
        """Analyze comparisons"""
        # Check for identity comparison with literals
        for op, comparator in zip(node.ops, node.comparators):
            if isinstance(op, (ast.Is, ast.IsNot)):
                if isinstance(comparator, (ast.Constant, ast.Num, ast.Str)):
                    self.issues.append(Issue(
                        file=self.filepath,
                        line=node.lineno,
                        column=node.col_offset,
                        severity="medium",
                        category="bug",
                        code="IDENTITY_COMPARISON",
                        message="Using 'is' for value comparison",
                        suggestion="Use '==' for value comparison, 'is' for identity"
                    ))

        self.generic_visit(node)

    def _calculate_complexity(self, node) -> int:
        """Calculate cyclomatic complexity"""
        complexity = 1
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
        return complexity

    def _is_in_with_context(self, node) -> bool:
        """Check if node is within a 'with' statement"""
        # This is a simplified check; full implementation would track AST context
        return False


class SecurityChecker:
    """Security-focused code checker"""

    @staticmethod
    def check_file(filepath: str, content: str) -> List[Issue]:
        """Check for security issues"""
        issues = []
        lines = content.split('\n')

        for i, line in enumerate(lines, 1):
            # Check for hardcoded passwords/secrets
            if any(keyword in line.lower() for keyword in ['password =', 'secret =', 'api_key =', 'token =']):
                if '"' in line or "'" in line:
                    issues.append(Issue(
                        file=filepath,
                        line=i,
                        column=0,
                        severity="critical",
                        category="security",
                        code="HARDCODED_SECRET",
                        message="Possible hardcoded secret detected",
                        suggestion="Use environment variables or secure config management"
                    ))

            # Check for weak hash algorithms
            if 'hashlib.md5' in line or 'hashlib.sha1' in line:
                issues.append(Issue(
                    file=filepath,
                    line=i,
                    column=0,
                    severity="high",
                    category="security",
                    code="WEAK_HASH",
                    message="Weak cryptographic hash algorithm (MD5/SHA1)",
                    suggestion="Use SHA256, SHA512, or bcrypt/argon2 for passwords"
                ))

            # Check for insecure random
            if 'random.random' in line or 'random.randint' in line:
                issues.append(Issue(
                    file=filepath,
                    line=i,
                    column=0,
                    severity="medium",
                    category="security",
                    code="INSECURE_RANDOM",
                    message="Using insecure random number generator",
                    suggestion="Use secrets module for security-sensitive operations"
                ))

        return issues


class PerformanceChecker:
    """Performance-focused code checker"""

    @staticmethod
    def check_file(filepath: str, content: str) -> List[Issue]:
        """Check for performance issues"""
        issues = []
        lines = content.split('\n')

        for i, line in enumerate(lines, 1):
            # Check for string concatenation in loops
            if ('for ' in line or 'while ' in line) and ('+=' in lines[min(i, len(lines)-1)]):
                if any('"' in lines[j] or "'" in lines[j] for j in range(i, min(i+5, len(lines)))):
                    issues.append(Issue(
                        file=filepath,
                        line=i,
                        column=0,
                        severity="medium",
                        category="performance",
                        code="STRING_CONCAT_LOOP",
                        message="String concatenation in loop",
                        suggestion="Use list and ''.join() for better performance"
                    ))

        return issues


class CodeAnalyzer:
    """Main code analyzer coordinating all checks"""

    def __init__(self, directory: str):
        self.directory = Path(directory)
        self.issues: List[Issue] = []

    def analyze(self) -> List[Issue]:
        """Analyze all Python files in directory"""
        python_files = list(self.directory.rglob('*.py'))

        for filepath in python_files:
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()

                # AST-based analysis
                analyzer = PythonAnalyzer(str(filepath))
                self.issues.extend(analyzer.analyze(content))

                # Security checks
                self.issues.extend(SecurityChecker.check_file(str(filepath), content))

                # Performance checks
                self.issues.extend(PerformanceChecker.check_file(str(filepath), content))

            except Exception as e:
                print(f"Error analyzing {filepath}: {e}", file=sys.stderr)

        return self.issues

    def generate_report(self, output_format: str = 'text') -> str:
        """Generate analysis report"""
        if output_format == 'json':
            return json.dumps([asdict(issue) for issue in self.issues], indent=2)

        # Text format
        report = []
        report.append("=" * 80)
        report.append("CODE ANALYSIS REPORT")
        report.append("=" * 80)
        report.append("")

        # Summary
        severity_counts = defaultdict(int)
        category_counts = defaultdict(int)
        for issue in self.issues:
            severity_counts[issue.severity] += 1
            category_counts[issue.category] += 1

        report.append("SUMMARY")
        report.append("-" * 80)
        report.append(f"Total issues: {len(self.issues)}")
        report.append(f"Critical: {severity_counts['critical']}")
        report.append(f"High: {severity_counts['high']}")
        report.append(f"Medium: {severity_counts['medium']}")
        report.append(f"Low: {severity_counts['low']}")
        report.append("")
        report.append("By Category:")
        for category, count in sorted(category_counts.items()):
            report.append(f"  {category}: {count}")
        report.append("")

        # Group by severity
        for severity in ['critical', 'high', 'medium', 'low']:
            severity_issues = [i for i in self.issues if i.severity == severity]
            if not severity_issues:
                continue

            report.append("=" * 80)
            report.append(f"{severity.upper()} ISSUES ({len(severity_issues)})")
            report.append("=" * 80)
            report.append("")

            for issue in severity_issues:
                report.append(f"[{issue.code}] {issue.message}")
                report.append(f"  File: {issue.file}:{issue.line}:{issue.column}")
                report.append(f"  Category: {issue.category}")
                if issue.suggestion:
                    report.append(f"  Suggestion: {issue.suggestion}")
                report.append("")

        return "\n".join(report)


def main():
    parser = argparse.ArgumentParser(description='Static Code Analyzer for Python')
    parser.add_argument('directory', help='Directory to analyze')
    parser.add_argument('--output', '-o', help='Output file (default: stdout)')
    parser.add_argument('--format', '-f', choices=['text', 'json'], default='text',
                       help='Output format')

    args = parser.parse_args()

    if not os.path.isdir(args.directory):
        print(f"Error: {args.directory} is not a directory", file=sys.stderr)
        sys.exit(1)

    print(f"Analyzing directory: {args.directory}")
    analyzer = CodeAnalyzer(args.directory)
    analyzer.analyze()

    report = analyzer.generate_report(args.format)

    if args.output:
        with open(args.output, 'w') as f:
            f.write(report)
        print(f"Report written to: {args.output}")
    else:
        print(report)


if __name__ == '__main__':
    main()
