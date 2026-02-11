# Common Weakness Enumeration (CWE) Quick Reference

Essential CWEs every Python developer should know.

## CWE-20: Improper Input Validation

**Description**: Not validating input properly allows malicious data to enter the system.

**Python Example**:
```python
# ❌ Bad
def process_age(age_str):
    age = int(age_str)  # Crashes on non-numeric input
    return age

# ✅ Good
def process_age(age_str):
    try:
        age = int(age_str)
        if not (0 <= age <= 150):
            raise ValueError("Age out of valid range")
        return age
    except ValueError as e:
        raise ValueError(f"Invalid age: {e}")
```

---

## CWE-78: OS Command Injection

**Description**: Executing OS commands with unsanitized input.

**Python Example**:
```python
# ❌ Bad - Command injection vulnerability
import os
filename = request.args.get('file')
os.system(f"cat {filename}")  # Vulnerable!

# ✅ Good - Use subprocess with list arguments
import subprocess
result = subprocess.run(['cat', filename], capture_output=True, shell=False)
```

---

## CWE-89: SQL Injection

**Description**: Constructing SQL queries from untrusted input.

**Python Example**:
```python
# ❌ Bad
query = f"SELECT * FROM users WHERE name = '{user_input}'"
cursor.execute(query)

# ✅ Good - Parameterized queries
query = "SELECT * FROM users WHERE name = ?"
cursor.execute(query, (user_input,))
```

---

## CWE-79: Cross-Site Scripting (XSS)

**Description**: Including untrusted data in web pages without proper escaping.

**Python Example (Flask)**:
```python
# ❌ Bad - Renders HTML directly
from flask import Markup
return Markup(user_input)  # XSS vulnerability!

# ✅ Good - Auto-escaping with Jinja2
return render_template('page.html', user_input=user_input)
# In template: {{ user_input }} is auto-escaped
```

---

## CWE-22: Path Traversal

**Description**: Accessing files outside intended directory.

**Python Example**:
```python
# ❌ Bad
import os
filename = request.args.get('file')
with open(f"/uploads/{filename}") as f:  # Can access ../../../etc/passwd
    content = f.read()

# ✅ Good - Validate path
from pathlib import Path

BASE_DIR = Path("/uploads").resolve()
filepath = (BASE_DIR / filename).resolve()

if not filepath.is_relative_to(BASE_DIR):
    raise ValueError("Invalid file path")

with open(filepath) as f:
    content = f.read()
```

---

## CWE-798: Hard-coded Credentials

**Description**: Storing passwords, API keys, or secrets in source code.

**Python Example**:
```python
# ❌ Bad
API_KEY = "sk-1234567890abcdef"  # Exposed in version control!
DATABASE_PASSWORD = "admin123"

# ✅ Good - Use environment variables
import os
API_KEY = os.getenv('API_KEY')
DATABASE_PASSWORD = os.getenv('DB_PASSWORD')

if not API_KEY:
    raise ValueError("API_KEY environment variable not set")
```

---

## CWE-327: Use of Broken Cryptographic Algorithm

**Description**: Using weak or broken cryptographic algorithms.

**Python Example**:
```python
# ❌ Bad - MD5 and SHA1 are broken
import hashlib
password_hash = hashlib.md5(password.encode()).hexdigest()

# ✅ Good - Use bcrypt for passwords
import bcrypt
password_hash = bcrypt.hashpw(password.encode(), bcrypt.gensalt())

# ✅ Good - Use SHA-256 for general hashing
file_hash = hashlib.sha256(file_data).hexdigest()
```

---

## CWE-502: Deserialization of Untrusted Data

**Description**: Deserializing data from untrusted sources can execute arbitrary code.

**Python Example**:
```python
# ❌ Bad - Pickle can execute code!
import pickle
data = pickle.loads(untrusted_input)

# ✅ Good - Use JSON for untrusted data
import json
data = json.loads(untrusted_input)

# ✅ Good - If pickle needed, verify signature
import hmac
import hashlib

def secure_pickle_loads(data, secret_key):
    signature = data[:32]
    pickled = data[32:]
    expected = hmac.new(secret_key, pickled, hashlib.sha256).digest()
    if not hmac.compare_digest(signature, expected):
        raise ValueError("Invalid signature")
    return pickle.loads(pickled)
```

---

## CWE-330: Use of Insufficiently Random Values

**Description**: Using weak random number generators for security purposes.

**Python Example**:
```python
# ❌ Bad - random module is NOT cryptographically secure
import random
session_token = ''.join(random.choices('0123456789abcdef', k=32))

# ✅ Good - Use secrets module
import secrets
session_token = secrets.token_hex(32)
api_key = secrets.token_urlsafe(32)
```

---

## CWE-119: Buffer Overflow (Memory Safety)

**Description**: Writing beyond allocated buffer (less common in Python but possible in C extensions).

**Python Mitigation**:
```python
# Python handles memory automatically, but be careful with:

# ❌ Potential issues with ctypes
import ctypes
buffer = ctypes.create_string_buffer(10)
# Writing more than 10 bytes can cause issues

# ✅ Use Python's native types when possible
# They handle bounds checking automatically
```

---

## CWE-400: Uncontrolled Resource Consumption

**Description**: Not limiting resource usage can lead to DoS.

**Python Example**:
```python
# ❌ Bad - No limits
def process_file(file):
    content = file.read()  # Could read gigabytes!
    return process(content)

# ✅ Good - Set limits
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB

def process_file(file):
    content = file.read(MAX_FILE_SIZE)
    if file.read(1):  # Check if there's more
        raise ValueError("File too large")
    return process(content)
```

---

## Quick Reference Table

| CWE | Name | Risk Level | Primary Defense |
|-----|------|------------|-----------------|
| CWE-20 | Input Validation | High | Validate all inputs |
| CWE-22 | Path Traversal | High | Validate file paths |
| CWE-78 | Command Injection | Critical | Never use `os.system()`, use `subprocess` with `shell=False` |
| CWE-79 | XSS | High | Auto-escape all outputs |
| CWE-89 | SQL Injection | Critical | Use parameterized queries |
| CWE-327 | Weak Crypto | High | Use bcrypt/SHA-256+ |
| CWE-330 | Weak Random | High | Use `secrets` module |
| CWE-400 | Resource Exhaustion | Medium | Set limits and timeouts |
| CWE-502 | Insecure Deserialization | Critical | Use JSON, not pickle |
| CWE-798 | Hardcoded Credentials | Critical | Use environment variables |

---

## Severity Levels

### Critical (Fix Immediately)
- CWE-78: Command Injection
- CWE-89: SQL Injection
- CWE-502: Insecure Deserialization
- CWE-798: Hardcoded Credentials

### High (Fix Within Days)
- CWE-20: Input Validation
- CWE-22: Path Traversal
- CWE-79: XSS
- CWE-327: Weak Cryptography
- CWE-330: Weak Random

### Medium (Fix Within Weeks)
- CWE-400: Resource Exhaustion
- CWE-209: Information Exposure
- CWE-352: CSRF

---

## Testing for CWEs

```python
# Example test for SQL injection (CWE-89)
def test_sql_injection_prevented():
    malicious_input = "admin' OR '1'='1"
    result = database.find_user(malicious_input)
    assert result is None  # Should not return all users

# Example test for command injection (CWE-78)
def test_command_injection_prevented():
    malicious_input = "file.txt; rm -rf /"
    with pytest.raises(ValueError):
        process_file(malicious_input)
```

---

## Resources

- [CWE Official Site](https://cwe.mitre.org/)
- [CWE Top 25](https://cwe.mitre.org/top25/)
- [MITRE ATT&CK](https://attack.mitre.org/)
- [OWASP Testing Guide](https://owasp.org/www-project-web-security-testing-guide/)
