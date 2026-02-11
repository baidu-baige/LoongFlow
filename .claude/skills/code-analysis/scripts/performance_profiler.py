#!/usr/bin/env python3
"""
Performance Profiler for Python Code
Identifies performance bottlenecks and provides optimization suggestions
Usage: python performance_profiler.py <file.py> [--threshold SECONDS]
"""

import ast
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass


@dataclass
class PerformanceIssue:
    """Performance issue with optimization suggestion"""
    line: int
    issue_type: str
    description: str
    current_complexity: str
    suggested_complexity: str
    suggestion: str
    code_snippet: str


class PerformanceAnalyzer(ast.NodeVisitor):
    """Analyzes code for performance issues"""

    def __init__(self, source_lines: List[str]):
        self.source_lines = source_lines
        self.issues: List[PerformanceIssue] = []
        self.loop_depth = 0

    def visit_For(self, node):
        """Analyze for loops"""
        self.loop_depth += 1

        # Check for nested loops
        if self.loop_depth >= 2:
            code = self._get_code_snippet(node.lineno, node.end_lineno or node.lineno + 1)
            self.issues.append(PerformanceIssue(
                line=node.lineno,
                issue_type="NESTED_LOOPS",
                description=f"Nested loop detected (depth: {self.loop_depth})",
                current_complexity=f"O(n^{self.loop_depth})",
                suggested_complexity="O(n) or O(n log n)",
                suggestion="Consider using dict/set for lookups, or list comprehension",
                code_snippet=code
            ))

        # Check for membership testing in list within loop
        for child in ast.walk(node):
            if isinstance(child, ast.Compare):
                for op in child.ops:
                    if isinstance(op, (ast.In, ast.NotIn)):
                        if isinstance(child.comparators[0], ast.Name):
                            code = self._get_code_snippet(child.lineno, child.lineno)
                            self.issues.append(PerformanceIssue(
                                line=child.lineno,
                                issue_type="LIST_MEMBERSHIP_IN_LOOP",
                                description="List membership test in loop (O(n) per iteration)",
                                current_complexity="O(n * m)",
                                suggested_complexity="O(n)",
                                suggestion="Convert list to set before loop for O(1) lookup",
                                code_snippet=code
                            ))

        self.generic_visit(node)
        self.loop_depth -= 1

    def visit_While(self, node):
        """Analyze while loops"""
        self.loop_depth += 1
        self.generic_visit(node)
        self.loop_depth -= 1

    def visit_ListComp(self, node):
        """Analyze list comprehensions"""
        # Check for nested list comprehensions
        nested_count = sum(1 for _ in ast.walk(node) if isinstance(_, ast.ListComp))
        if nested_count > 1:
            code = self._get_code_snippet(node.lineno, node.end_lineno or node.lineno)
            self.issues.append(PerformanceIssue(
                line=node.lineno,
                issue_type="NESTED_LIST_COMP",
                description="Nested list comprehension",
                current_complexity="O(n * m)",
                suggested_complexity="O(n) with generators or itertools",
                suggestion="Consider using itertools.chain or generator expressions",
                code_snippet=code
            ))

        self.generic_visit(node)

    def visit_Call(self, node):
        """Analyze function calls"""
        # Check for repeated function calls in loops
        if self.loop_depth > 0:
            if isinstance(node.func, ast.Attribute):
                if node.func.attr in ('append', 'extend', 'insert'):
                    code = self._get_code_snippet(node.lineno, node.lineno)
                    self.issues.append(PerformanceIssue(
                        line=node.lineno,
                        issue_type="REPEATED_APPEND",
                        description=f"Repeated {node.func.attr}() in loop",
                        current_complexity="O(n) with potential memory reallocation",
                        suggested_complexity="O(n) with list comprehension",
                        suggestion="Use list comprehension or pre-allocate list size",
                        code_snippet=code
                    ))

        # Check for inefficient string operations
        if isinstance(node.func, ast.Attribute):
            if node.func.attr in ('format', 'replace', 'join'):
                if self.loop_depth > 0:
                    code = self._get_code_snippet(node.lineno, node.lineno)
                    self.issues.append(PerformanceIssue(
                        line=node.lineno,
                        issue_type="STRING_OP_IN_LOOP",
                        description=f"String {node.func.attr}() in loop",
                        current_complexity="O(n * m) - string operations are expensive",
                        suggested_complexity="O(n) - minimize string operations",
                        suggestion="Build list and join once, or use string builder",
                        code_snippet=code
                    ))

        self.generic_visit(node)

    def visit_FunctionDef(self, node):
        """Analyze function definitions"""
        # Check for unnecessary global lookups
        global_lookups = []
        for child in ast.walk(node):
            if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load):
                if child.id in ('len', 'range', 'enumerate', 'zip'):
                    global_lookups.append((child.lineno, child.id))

        if len(global_lookups) > 3:  # Arbitrary threshold
            unique_funcs = set(func for _, func in global_lookups)
            code = self._get_code_snippet(node.lineno, min(node.lineno + 3, len(self.source_lines)))
            self.issues.append(PerformanceIssue(
                line=node.lineno,
                issue_type="REPEATED_GLOBAL_LOOKUP",
                description=f"Repeated global lookups: {', '.join(unique_funcs)}",
                current_complexity="O(n) global dict lookups",
                suggested_complexity="O(1) local variable",
                suggestion="Cache global functions as local variables at function start",
                code_snippet=code
            ))

        self.generic_visit(node)

    def _get_code_snippet(self, start: int, end: int) -> str:
        """Get code snippet from source"""
        start_idx = max(0, start - 1)
        end_idx = min(len(self.source_lines), end)
        return '\n'.join(self.source_lines[start_idx:end_idx])


class AlgorithmicComplexityAnalyzer:
    """Analyzes algorithmic complexity patterns"""

    @staticmethod
    def analyze(source_code: str, source_lines: List[str]) -> List[PerformanceIssue]:
        """Detect common algorithmic anti-patterns"""
        issues = []

        # Pattern: list.sort() or sorted() in loop
        if 'for ' in source_code and ('sort()' in source_code or 'sorted(' in source_code):
            for i, line in enumerate(source_lines, 1):
                if 'sort' in line and any('for ' in source_lines[j] for j in range(max(0, i-5), i)):
                    issues.append(PerformanceIssue(
                        line=i,
                        issue_type="SORT_IN_LOOP",
                        description="Sorting operation inside loop",
                        current_complexity="O(n * m log m)",
                        suggested_complexity="O(m log m + n)",
                        suggestion="Sort once before loop, not in each iteration",
                        code_snippet=line.strip()
                    ))

        # Pattern: dict.keys() in loop
        for i, line in enumerate(source_lines, 1):
            if '.keys()' in line and ('for ' in line or any('for ' in source_lines[j] for j in range(max(0, i-3), i))):
                issues.append(PerformanceIssue(
                    line=i,
                    issue_type="UNNECESSARY_KEYS",
                    description="Unnecessary .keys() call when iterating dict",
                    current_complexity="O(n) with extra memory",
                    suggested_complexity="O(n) without extra memory",
                    suggestion="Iterate dict directly: 'for key in dict:' instead of 'for key in dict.keys():'",
                    code_snippet=line.strip()
                ))

        # Pattern: list(...) when not needed
        for i, line in enumerate(source_lines, 1):
            if 'list(' in line and ('range(' in line or 'map(' in line or 'filter(' in line):
                issues.append(PerformanceIssue(
                    line=i,
                    issue_type="UNNECESSARY_LIST",
                    description="Unnecessary list() conversion",
                    current_complexity="O(n) memory + O(n) time",
                    suggested_complexity="O(1) memory with lazy iteration",
                    suggestion="Use generators/iterators directly unless you need materialized list",
                    code_snippet=line.strip()
                ))

        return issues


def analyze_file(filepath: str, threshold_complexity: int = 2) -> Tuple[List[PerformanceIssue], Dict[str, int]]:
    """Analyze a Python file for performance issues"""
    with open(filepath, 'r', encoding='utf-8') as f:
        source_code = f.read()
        source_lines = source_code.split('\n')

    # AST-based analysis
    tree = ast.parse(source_code)
    analyzer = PerformanceAnalyzer(source_lines)
    analyzer.visit(tree)
    issues = analyzer.issues

    # Pattern-based analysis
    issues.extend(AlgorithmicComplexityAnalyzer.analyze(source_code, source_lines))

    # Statistics
    stats = {
        'total_issues': len(issues),
        'nested_loops': sum(1 for i in issues if i.issue_type == 'NESTED_LOOPS'),
        'list_operations': sum(1 for i in issues if 'LIST' in i.issue_type),
        'string_operations': sum(1 for i in issues if 'STRING' in i.issue_type),
    }

    return issues, stats


def generate_report(filepath: str, issues: List[PerformanceIssue], stats: Dict[str, int]) -> str:
    """Generate performance analysis report"""
    report = []
    report.append("=" * 80)
    report.append("PERFORMANCE ANALYSIS REPORT")
    report.append("=" * 80)
    report.append(f"File: {filepath}")
    report.append("")

    # Summary
    report.append("SUMMARY")
    report.append("-" * 80)
    report.append(f"Total performance issues: {stats['total_issues']}")
    report.append(f"Nested loops (O(n²) or worse): {stats['nested_loops']}")
    report.append(f"Inefficient list operations: {stats['list_operations']}")
    report.append(f"String operation issues: {stats['string_operations']}")
    report.append("")

    if not issues:
        report.append("✓ No performance issues detected!")
        return "\n".join(report)

    # Detailed issues
    report.append("DETAILED ISSUES")
    report.append("=" * 80)

    for idx, issue in enumerate(sorted(issues, key=lambda x: x.line), 1):
        report.append(f"\nIssue #{idx}: {issue.issue_type}")
        report.append("-" * 80)
        report.append(f"Line: {issue.line}")
        report.append(f"Description: {issue.description}")
        report.append(f"Current Complexity: {issue.current_complexity}")
        report.append(f"Optimized Complexity: {issue.suggested_complexity}")
        report.append(f"\nCode:")
        for line in issue.code_snippet.split('\n'):
            report.append(f"  {line}")
        report.append(f"\n💡 Suggestion: {issue.suggestion}")
        report.append("")

    # Optimization recommendations
    report.append("=" * 80)
    report.append("OPTIMIZATION RECOMMENDATIONS")
    report.append("=" * 80)
    report.append("")
    report.append("1. Address nested loops first - they have the biggest impact")
    report.append("2. Convert lists to sets/dicts for membership testing")
    report.append("3. Use list comprehensions instead of repeated append()")
    report.append("4. Minimize string operations in loops")
    report.append("5. Cache global function lookups as local variables")
    report.append("6. Use generators for large datasets to save memory")
    report.append("")

    return "\n".join(report)


def main():
    parser = argparse.ArgumentParser(description='Performance Profiler for Python')
    parser.add_argument('file', help='Python file to analyze')
    parser.add_argument('--threshold', type=int, default=2,
                       help='Complexity threshold (default: 2)')
    parser.add_argument('--output', '-o', help='Output file (default: stdout)')

    args = parser.parse_args()

    if not Path(args.file).exists():
        print(f"Error: File not found: {args.file}", file=sys.stderr)
        sys.exit(1)

    try:
        issues, stats = analyze_file(args.file, args.threshold)
        report = generate_report(args.file, issues, stats)

        if args.output:
            with open(args.output, 'w') as f:
                f.write(report)
            print(f"Report written to: {args.output}")
        else:
            print(report)

    except SyntaxError as e:
        print(f"Syntax error in {args.file}: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error analyzing file: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
