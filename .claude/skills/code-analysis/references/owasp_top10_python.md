# OWASP Top 10 for Python Applications

This document provides practical guidance on addressing the OWASP Top 10 security risks in Python applications.

## A01:2021 – Broken Access Control

### Risk
Users can act outside of their intended permissions, accessing data or functionality they shouldn't.

### Python Examples

**❌ Bad**:
```python
@app.route('/user/<int:user_id>/profile')
def view_profile(user_id):
    user = User.query.get(user_id)
    return render_template('profile.html', user=user)
```

**✅ Good**:
```python
from flask_login import current_user, login_required

@app.route('/user/<int:user_id>/profile')
@login_required
def view_profile(user_id):
    if current_user.id != user_id and not current_user.is_admin:
        abort(403)  # Forbidden
    user = User.query.get_or_404(user_id)
    return render_template('profile.html', user=user)
```

### Remediation
- Implement proper authorization checks
- Use role-based access control (RBAC)
- Deny by default
- Log access control failures

---

## A02:2021 – Cryptographic Failures

### Risk
Sensitive data exposed due to weak or missing encryption.

### Python Examples

**❌ Bad**:
```python
import hashlib

# MD5 is cryptographically broken
password_hash = hashlib.md5(password.encode()).hexdigest()

# Plain text password storage
user.password = password
```

**✅ Good**:
```python
import bcrypt

# Strong password hashing
password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

# Verify password
if bcrypt.checkpw(password.encode('utf-8'), stored_hash):
    # Password correct
    pass
```

### Remediation
- Use bcrypt, argon2, or scrypt for passwords
- Use SHA-256 or SHA-512 for general hashing
- Never store passwords in plain text
- Use TLS 1.2+ for data in transit

---

## A03:2021 – Injection

### Risk
Untrusted data sent to an interpreter as part of a command or query.

### SQL Injection

**❌ Bad**:
```python
# String formatting - vulnerable!
query = f"SELECT * FROM users WHERE username = '{username}'"
cursor.execute(query)

# String concatenation - vulnerable!
query = "SELECT * FROM users WHERE id = " + user_id
cursor.execute(query)
```

**✅ Good**:
```python
# Parameterized query - safe
query = "SELECT * FROM users WHERE username = ?"
cursor.execute(query, (username,))

# Or with named parameters
query = "SELECT * FROM users WHERE username = :username"
cursor.execute(query, {'username': username})
```

### Command Injection

**❌ Bad**:
```python
import os

# Vulnerable to command injection
os.system(f"ping {user_input}")
```

**✅ Good**:
```python
import subprocess

# Safe: shell=False and list arguments
subprocess.run(['ping', '-c', '1', user_input], shell=False, capture_output=True)
```

### Code Injection

**❌ Bad**:
```python
# Never use eval with user input!
result = eval(user_formula)
```

**✅ Good**:
```python
import ast

# Safe evaluation of literals only
try:
    result = ast.literal_eval(user_input)
except (ValueError, SyntaxError):
    # Invalid input
    pass
```

---

## A04:2021 – Insecure Design

### Risk
Missing or ineffective security controls in the design phase.

### Examples

**Design Principles**:
1. **Principle of Least Privilege**: Grant minimum required permissions
2. **Defense in Depth**: Multiple layers of security
3. **Fail Securely**: Default to deny access on error
4. **Separation of Duties**: No single person has complete control

**Python Implementation**:
```python
class SecureUserService:
    def __init__(self, db, audit_log, permission_checker):
        self.db = db
        self.audit_log = audit_log
        self.permission_checker = permission_checker

    def delete_user(self, requester_id: int, user_id: int) -> bool:
        # Check permission
        if not self.permission_checker.can_delete_user(requester_id):
            self.audit_log.log_failed_attempt(requester_id, 'delete_user', user_id)
            raise PermissionError("Insufficient privileges")

        # Additional check: can't delete yourself
        if requester_id == user_id:
            raise ValueError("Cannot delete own account")

        # Perform deletion
        try:
            self.db.delete_user(user_id)
            self.audit_log.log_success(requester_id, 'delete_user', user_id)
            return True
        except Exception as e:
            self.audit_log.log_error(requester_id, 'delete_user', user_id, str(e))
            raise
```

---

## A05:2021 – Security Misconfiguration

### Risk
Insecure default configurations, incomplete setups, open cloud storage, verbose error messages.

### Python Examples

**❌ Bad**:
```python
# Debug mode in production
app.config['DEBUG'] = True

# Default secret key
app.config['SECRET_KEY'] = 'default-secret-key'

# Disabled SSL verification
requests.get(url, verify=False)
```

**✅ Good**:
```python
import os
import secrets

# Environment-based configuration
app.config['DEBUG'] = os.getenv('FLASK_ENV') == 'development'

# Strong random secret key
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY') or secrets.token_hex(32)

# Enable SSL verification
requests.get(url, verify=True, timeout=30)
```

---

## A06:2021 – Vulnerable and Outdated Components

### Risk
Using components with known vulnerabilities.

### Python Best Practices

**Dependency Management**:
```bash
# Keep dependencies updated
pip list --outdated

# Check for known vulnerabilities
pip-audit

# Or use safety
safety check

# Pin specific versions in requirements.txt
Flask==2.3.0  # Not Flask>=2.0.0
```

**requirements.txt Example**:
```text
# Pin exact versions for reproducibility
Flask==2.3.0
SQLAlchemy==2.0.15
cryptography==41.0.0

# Avoid wildcards or overly broad ranges
# Bad: Flask>=2.0.0
# Bad: Flask==2.*
```

---

## A07:2021 – Identification and Authentication Failures

### Risk
Broken authentication mechanisms allowing attackers to compromise passwords, keys, or session tokens.

### Python Examples

**Password Policies**:
```python
import re
import bcrypt

def validate_password(password: str) -> tuple[bool, str]:
    """Validate password strength"""
    if len(password) < 12:
        return False, "Password must be at least 12 characters"

    if not re.search(r'[A-Z]', password):
        return False, "Password must contain uppercase letter"

    if not re.search(r'[a-z]', password):
        return False, "Password must contain lowercase letter"

    if not re.search(r'\d', password):
        return False, "Password must contain digit"

    if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
        return False, "Password must contain special character"

    return True, "Password is strong"


def hash_password(password: str) -> bytes:
    """Securely hash password"""
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt(rounds=12))
```

**Session Management**:
```python
from flask import session
import secrets

# Generate secure session token
session_token = secrets.token_urlsafe(32)

# Set secure cookie flags
app.config.update(
    SESSION_COOKIE_SECURE=True,  # HTTPS only
    SESSION_COOKIE_HTTPONLY=True,  # No JavaScript access
    SESSION_COOKIE_SAMESITE='Lax',  # CSRF protection
    PERMANENT_SESSION_LIFETIME=1800  # 30 minutes
)
```

**Rate Limiting**:
```python
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

limiter = Limiter(
    app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"]
)

@app.route('/login', methods=['POST'])
@limiter.limit("5 per minute")  # Prevent brute force
def login():
    # Login logic
    pass
```

---

## A08:2021 – Software and Data Integrity Failures

### Risk
Code and infrastructure that does not protect against integrity violations.

### Python Examples

**Insecure Deserialization**:

**❌ Bad**:
```python
import pickle

# Pickle can execute arbitrary code!
data = pickle.loads(untrusted_data)
```

**✅ Good**:
```python
import json
import hmac
import hashlib

def serialize_data(data: dict, secret_key: bytes) -> bytes:
    """Serialize with integrity check"""
    json_data = json.dumps(data).encode('utf-8')
    signature = hmac.new(secret_key, json_data, hashlib.sha256).digest()
    return signature + json_data


def deserialize_data(serialized: bytes, secret_key: bytes) -> dict:
    """Deserialize with integrity verification"""
    signature = serialized[:32]
    json_data = serialized[32:]

    expected_sig = hmac.new(secret_key, json_data, hashlib.sha256).digest()
    if not hmac.compare_digest(signature, expected_sig):
        raise ValueError("Data integrity check failed")

    return json.loads(json_data)
```

---

## A09:2021 – Security Logging and Monitoring Failures

### Risk
Without proper logging and monitoring, breaches cannot be detected.

### Python Implementation

```python
import logging
import json
from datetime import datetime

# Configure security logging
security_logger = logging.getLogger('security')
security_logger.setLevel(logging.INFO)

handler = logging.FileHandler('security.log')
handler.setFormatter(logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
))
security_logger.addHandler(handler)


class SecurityAudit:
    """Security audit logging"""

    @staticmethod
    def log_authentication(user_id: int, success: bool, ip_address: str):
        """Log authentication attempt"""
        security_logger.info(json.dumps({
            'event': 'authentication',
            'user_id': user_id,
            'success': success,
            'ip_address': ip_address,
            'timestamp': datetime.utcnow().isoformat()
        }))

    @staticmethod
    def log_authorization_failure(user_id: int, resource: str, action: str):
        """Log authorization failure"""
        security_logger.warning(json.dumps({
            'event': 'authorization_failure',
            'user_id': user_id,
            'resource': resource,
            'action': action,
            'timestamp': datetime.utcnow().isoformat()
        }))

    @staticmethod
    def log_data_access(user_id: int, data_type: str, record_id: int):
        """Log sensitive data access"""
        security_logger.info(json.dumps({
            'event': 'data_access',
            'user_id': user_id,
            'data_type': data_type,
            'record_id': record_id,
            'timestamp': datetime.utcnow().isoformat()
        }))
```

---

## A10:2021 – Server-Side Request Forgery (SSRF)

### Risk
Fetching a remote resource without validating the user-supplied URL.

### Python Examples

**❌ Bad**:
```python
import requests

@app.route('/fetch')
def fetch_url():
    url = request.args.get('url')
    # Vulnerable - can access internal services!
    response = requests.get(url)
    return response.content
```

**✅ Good**:
```python
import requests
from urllib.parse import urlparse
import ipaddress

ALLOWED_DOMAINS = ['api.example.com', 'cdn.example.com']

def is_safe_url(url: str) -> bool:
    """Validate URL is safe to fetch"""
    try:
        parsed = urlparse(url)

        # Only allow HTTPS
        if parsed.scheme != 'https':
            return False

        # Check domain whitelist
        if parsed.hostname not in ALLOWED_DOMAINS:
            return False

        # Prevent access to private IPs
        try:
            ip = ipaddress.ip_address(parsed.hostname)
            if ip.is_private or ip.is_loopback:
                return False
        except ValueError:
            pass  # Not an IP address

        return True
    except Exception:
        return False


@app.route('/fetch')
def fetch_url():
    url = request.args.get('url')

    if not is_safe_url(url):
        abort(400, "Invalid URL")

    response = requests.get(url, timeout=5, allow_redirects=False)
    return response.content
```

---

## General Security Checklist

### Input Validation
- [ ] Validate all user inputs
- [ ] Use allowlists, not denylists
- [ ] Sanitize data for output context (HTML, SQL, etc.)

### Authentication & Authorization
- [ ] Use strong password hashing (bcrypt, argon2)
- [ ] Implement rate limiting
- [ ] Use secure session management
- [ ] Check authorization on every request

### Data Protection
- [ ] Use HTTPS everywhere
- [ ] Encrypt sensitive data at rest
- [ ] Use environment variables for secrets
- [ ] Implement proper key management

### Error Handling
- [ ] Don't expose stack traces to users
- [ ] Log errors securely
- [ ] Fail securely (deny by default)

### Dependencies
- [ ] Keep all dependencies updated
- [ ] Use `pip-audit` or `safety check`
- [ ] Pin dependency versions

### Monitoring
- [ ] Log security events
- [ ] Monitor for suspicious patterns
- [ ] Set up alerts for anomalies

---

## Resources

- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [OWASP Cheat Sheet Series](https://cheatsheetseries.owasp.org/)
- [CWE Top 25](https://cwe.mitre.org/top25/)
- [Python Security Best Practices](https://python.readthedocs.io/en/latest/library/security_warnings.html)
