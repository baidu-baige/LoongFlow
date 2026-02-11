#!/usr/bin/env python3
"""
Security Scanner for Python Code
Detects common security vulnerabilities and provides remediation advice
Usage: python security_scanner.py <directory> [--output OUTPUT]
"""

import re
import ast
import sys
import os
import argparse
import json
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass, asdict


@dataclass
class SecurityVulnerability:
    """Security vulnerability finding"""
    file: str
    line: int
    severity: str  # critical, high, medium, low
    vulnerability_type: str
    cwe_id: str  # Common Weakness Enumeration ID
    description: str
    evidence: str
    remediation: str
    references: List[str]


class SecurityScanner:
    """Comprehensive security scanner"""

    def __init__(self):
        self.vulnerabilities: List[SecurityVulnerability] = []

        # Dangerous function patterns
        self.dangerous_functions = {
            'eval': ('CWE-95', 'Code Injection'),
            'exec': ('CWE-95', 'Code Injection'),
            'compile': ('CWE-95', 'Code Injection'),
            '__import__': ('CWE-95', 'Dynamic Import'),
            'pickle.loads': ('CWE-502', 'Deserialization of Untrusted Data'),
            'yaml.load': ('CWE-502', 'Unsafe YAML Deserialization'),
            'subprocess.call': ('CWE-78', 'OS Command Injection'),
            'os.system': ('CWE-78', 'OS Command Injection'),
        }

        # Insecure patterns
        self.insecure_patterns = [
            (r'password\s*=\s*["\'].*["\']', 'CWE-798', 'Hardcoded Password'),
            (r'api[_-]?key\s*=\s*["\'].*["\']', 'CWE-798', 'Hardcoded API Key'),
            (r'secret\s*=\s*["\'].*["\']', 'CWE-798', 'Hardcoded Secret'),
            (r'token\s*=\s*["\'].*["\']', 'CWE-798', 'Hardcoded Token'),
            (r'aws[_-]?access[_-]?key', 'CWE-798', 'AWS Credentials'),
            (r'hashlib\.md5', 'CWE-327', 'Weak Hash Algorithm (MD5)'),
            (r'hashlib\.sha1', 'CWE-327', 'Weak Hash Algorithm (SHA1)'),
            (r'random\.random', 'CWE-330', 'Weak Random Number Generator'),
            (r'urllib\.request\.urlopen', 'CWE-918', 'SSRF via URL Open'),
        ]

    def scan_file(self, filepath: str) -> None:
        """Scan a single Python file"""
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            lines = content.split('\n')

        # AST-based analysis
        try:
            tree = ast.parse(content)
            self._analyze_ast(tree, filepath, lines)
        except SyntaxError:
            pass  # Skip files with syntax errors

        # Pattern-based analysis
        self._analyze_patterns(filepath, lines)

    def _analyze_ast(self, tree: ast.AST, filepath: str, lines: List[str]) -> None:
        """AST-based security analysis"""

        for node in ast.walk(tree):
            # Check for dangerous function calls
            if isinstance(node, ast.Call):
                func_name = self._get_func_name(node.func)
                if func_name in self.dangerous_functions:
                    cwe_id, vuln_type = self.dangerous_functions[func_name]
                    evidence = lines[node.lineno - 1].strip() if node.lineno <= len(lines) else ""

                    self.vulnerabilities.append(SecurityVulnerability(
                        file=filepath,
                        line=node.lineno,
                        severity='critical',
                        vulnerability_type=vuln_type,
                        cwe_id=cwe_id,
                        description=f"Use of dangerous function: {func_name}()",
                        evidence=evidence,
                        remediation=self._get_remediation(func_name),
                        references=[
                            f"https://cwe.mitre.org/data/definitions/{cwe_id.split('-')[1]}.html"
                        ]
                    ))

                # Check for SQL injection
                if func_name == 'execute':
                    for arg in node.args:
                        if self._is_string_concat(arg):
                            evidence = lines[node.lineno - 1].strip() if node.lineno <= len(lines) else ""
                            self.vulnerabilities.append(SecurityVulnerability(
                                file=filepath,
                                line=node.lineno,
                                severity='critical',
                                vulnerability_type='SQL Injection',
                                cwe_id='CWE-89',
                                description="Possible SQL injection via string concatenation",
                                evidence=evidence,
                                remediation="Use parameterized queries with placeholders (?, %s) instead of string formatting",
                                references=[
                                    "https://cwe.mitre.org/data/definitions/89.html",
                                    "https://cheatsheetseries.owasp.org/cheatsheets/SQL_Injection_Prevention_Cheat_Sheet.html"
                                ]
                            ))

            # Check for assert usage (disabled in production)
            if isinstance(node, ast.Assert):
                evidence = lines[node.lineno - 1].strip() if node.lineno <= len(lines) else ""
                self.vulnerabilities.append(SecurityVulnerability(
                    file=filepath,
                    line=node.lineno,
                    severity='medium',
                    vulnerability_type='Assert in Production Code',
                    cwe_id='CWE-670',
                    description="Assert statement (disabled in Python -O mode)",
                    evidence=evidence,
                    remediation="Use proper error handling with exceptions instead of assert",
                    references=["https://docs.python.org/3/reference/simple_stmts.html#assert"]
                ))

            # Check for bare except
            if isinstance(node, ast.ExceptHandler):
                if node.type is None:
                    evidence = lines[node.lineno - 1].strip() if node.lineno <= len(lines) else ""
                    self.vulnerabilities.append(SecurityVulnerability(
                        file=filepath,
                        line=node.lineno,
                        severity='low',
                        vulnerability_type='Overly Broad Exception Handling',
                        cwe_id='CWE-396',
                        description="Bare except clause catches all exceptions including SystemExit",
                        evidence=evidence,
                        remediation="Catch specific exception types instead of bare except",
                        references=["https://cwe.mitre.org/data/definitions/396.html"]
                    ))

    def _analyze_patterns(self, filepath: str, lines: List[str]) -> None:
        """Pattern-based security analysis"""
        for i, line in enumerate(lines, 1):
            for pattern, cwe_id, vuln_type in self.insecure_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    self.vulnerabilities.append(SecurityVulnerability(
                        file=filepath,
                        line=i,
                        severity=self._get_severity_for_cwe(cwe_id),
                        vulnerability_type=vuln_type,
                        cwe_id=cwe_id,
                        description=f"Detected: {vuln_type}",
                        evidence=line.strip(),
                        remediation=self._get_remediation_for_pattern(vuln_type),
                        references=[
                            f"https://cwe.mitre.org/data/definitions/{cwe_id.split('-')[1]}.html"
                        ]
                    ))

            # Check for insecure SSL/TLS
            if 'verify=False' in line or 'CERT_NONE' in line:
                self.vulnerabilities.append(SecurityVulnerability(
                    file=filepath,
                    line=i,
                    severity='high',
                    vulnerability_type='Disabled Certificate Verification',
                    cwe_id='CWE-295',
                    description="SSL/TLS certificate verification disabled",
                    evidence=line.strip(),
                    remediation="Enable certificate verification (remove verify=False)",
                    references=[
                        "https://cwe.mitre.org/data/definitions/295.html",
                        "https://cheatsheetseries.owasp.org/cheatsheets/Transport_Layer_Protection_Cheat_Sheet.html"
                    ]
                ))

            # Check for debug mode in production indicators
            if 'DEBUG = True' in line or 'debug=True' in line:
                self.vulnerabilities.append(SecurityVulnerability(
                    file=filepath,
                    line=i,
                    severity='high',
                    vulnerability_type='Debug Mode Enabled',
                    cwe_id='CWE-489',
                    description="Debug mode may expose sensitive information",
                    evidence=line.strip(),
                    remediation="Disable debug mode in production (use environment variables)",
                    references=["https://cwe.mitre.org/data/definitions/489.html"]
                ))

    def _get_func_name(self, node: ast.AST) -> str:
        """Extract function name from AST node"""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            value = self._get_func_name(node.value)
            return f"{value}.{node.attr}" if value else node.attr
        return ""

    def _is_string_concat(self, node: ast.AST) -> bool:
        """Check if node is string concatenation"""
        return isinstance(node, (ast.JoinedStr, ast.BinOp, ast.Call)) and \
               (isinstance(node, ast.Call) and
                isinstance(node.func, ast.Attribute) and
                node.func.attr == 'format')

    def _get_severity_for_cwe(self, cwe_id: str) -> str:
        """Map CWE to severity"""
        critical_cwes = ['CWE-89', 'CWE-95', 'CWE-502', 'CWE-798']
        high_cwes = ['CWE-78', 'CWE-295', 'CWE-327', 'CWE-489']

        if cwe_id in critical_cwes:
            return 'critical'
        elif cwe_id in high_cwes:
            return 'high'
        else:
            return 'medium'

    def _get_remediation(self, func_name: str) -> str:
        """Get remediation advice for dangerous function"""
        remediations = {
            'eval': "Avoid eval(). Use ast.literal_eval() for safe evaluation of literals, or json.loads() for JSON",
            'exec': "Avoid exec(). Redesign to use functions, classes, or configuration files instead",
            'compile': "Avoid compile() with untrusted input. Validate and sanitize all inputs",
            '__import__': "Use importlib.import_module() and validate module names against whitelist",
            'pickle.loads': "Use json.loads() for untrusted data. If pickle needed, verify data source and use hmac signature",
            'yaml.load': "Use yaml.safe_load() instead of yaml.load() to prevent arbitrary code execution",
            'subprocess.call': "Use subprocess.run() with shell=False and list arguments (not string). Validate all inputs",
            'os.system': "Use subprocess.run() with shell=False instead. Never pass unsanitized user input",
        }
        return remediations.get(func_name, "Avoid this function or carefully validate all inputs")

    def _get_remediation_for_pattern(self, vuln_type: str) -> str:
        """Get remediation for pattern-based vulnerability"""
        if 'Hardcoded' in vuln_type:
            return "Use environment variables or secure config management (e.g., AWS Secrets Manager, HashiCorp Vault)"
        elif 'Weak Hash' in vuln_type:
            return "Use SHA-256, SHA-512 for general hashing. Use bcrypt, argon2, or scrypt for passwords"
        elif 'Weak Random' in vuln_type:
            return "Use secrets module for cryptographic purposes: secrets.token_bytes(), secrets.token_hex()"
        return "Follow security best practices and OWASP guidelines"

    def scan_directory(self, directory: str) -> None:
        """Scan all Python files in directory"""
        for filepath in Path(directory).rglob('*.py'):
            self.scan_file(str(filepath))

    def generate_report(self, output_format: str = 'text') -> str:
        """Generate security report"""
        if output_format == 'json':
            return json.dumps([asdict(v) for v in self.vulnerabilities], indent=2)

        # Text report
        report = []
        report.append("=" * 90)
        report.append("SECURITY SCAN REPORT")
        report.append("=" * 90)
        report.append("")

        # Summary
        severity_counts = {'critical': 0, 'high': 0, 'medium': 0, 'low': 0}
        for vuln in self.vulnerabilities:
            severity_counts[vuln.severity] += 1

        report.append("EXECUTIVE SUMMARY")
        report.append("-" * 90)
        report.append(f"Total vulnerabilities found: {len(self.vulnerabilities)}")
        report.append(f"  🔴 Critical: {severity_counts['critical']}")
        report.append(f"  🟠 High: {severity_counts['high']}")
        report.append(f"  🟡 Medium: {severity_counts['medium']}")
        report.append(f"  🔵 Low: {severity_counts['low']}")
        report.append("")

        if severity_counts['critical'] > 0:
            report.append("⚠️  CRITICAL vulnerabilities found - immediate action required!")
            report.append("")

        # Detailed findings by severity
        for severity in ['critical', 'high', 'medium', 'low']:
            vulns = [v for v in self.vulnerabilities if v.severity == severity]
            if not vulns:
                continue

            icon = {'critical': '🔴', 'high': '🟠', 'medium': '🟡', 'low': '🔵'}[severity]
            report.append("=" * 90)
            report.append(f"{icon} {severity.upper()} SEVERITY ({len(vulns)} findings)")
            report.append("=" * 90)
            report.append("")

            for vuln in vulns:
                report.append(f"[{vuln.cwe_id}] {vuln.vulnerability_type}")
                report.append(f"  File: {vuln.file}:{vuln.line}")
                report.append(f"  Description: {vuln.description}")
                report.append(f"  Evidence: {vuln.evidence}")
                report.append(f"  🔧 Remediation: {vuln.remediation}")
                if vuln.references:
                    report.append(f"  📚 References:")
                    for ref in vuln.references:
                        report.append(f"     {ref}")
                report.append("")

        # Recommendations
        report.append("=" * 90)
        report.append("SECURITY RECOMMENDATIONS")
        report.append("=" * 90)
        report.append("")
        report.append("1. Fix all CRITICAL vulnerabilities immediately")
        report.append("2. Address HIGH severity issues within 1 week")
        report.append("3. Plan remediation for MEDIUM severity issues")
        report.append("4. Implement security code review process")
        report.append("5. Enable automated security scanning in CI/CD")
        report.append("6. Follow OWASP Top 10 guidelines")
        report.append("7. Conduct regular security training for developers")
        report.append("")

        return "\n".join(report)


def main():
    parser = argparse.ArgumentParser(description='Security Scanner for Python')
    parser.add_argument('directory', help='Directory to scan')
    parser.add_argument('--output', '-o', help='Output file (default: stdout)')
    parser.add_argument('--format', '-f', choices=['text', 'json'], default='text')

    args = parser.parse_args()

    if not os.path.isdir(args.directory):
        print(f"Error: {args.directory} is not a directory", file=sys.stderr)
        sys.exit(1)

    print(f"🔍 Scanning directory: {args.directory}")
    scanner = SecurityScanner()
    scanner.scan_directory(args.directory)

    report = scanner.generate_report(args.format)

    if args.output:
        with open(args.output, 'w') as f:
            f.write(report)
        print(f"✓ Report written to: {args.output}")
    else:
        print(report)

    # Exit with error code if critical vulns found
    critical_count = sum(1 for v in scanner.vulnerabilities if v.severity == 'critical')
    if critical_count > 0:
        sys.exit(1)


if __name__ == '__main__':
    main()
