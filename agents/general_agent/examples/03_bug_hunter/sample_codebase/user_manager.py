"""
User Management System
A simple system for managing user accounts and authentication
"""

import json
import hashlib


class UserManager:
    """Manages user accounts"""

    def __init__(self, data_file="users.json"):
        self.data_file = data_file
        self.users = []
        self.load_users()

    def load_users(self):
        """Load users from JSON file"""
        f = open(self.data_file, 'r')
        self.users = json.load(f)
        # Missing: f.close()

    def save_users(self):
        """Save users to JSON file"""
        with open(self.data_file, 'w') as f:
            json.dump(self.users, f)

    def create_user(self, username, password, email):
        """Create a new user account"""
        user = {
            'username': username,
            'password': password,
            'email': email,
            'active': True
        }
        self.users.append(user)
        self.save_users()
        return user

    def find_user(self, username):
        """Find user by username"""
        for user in self.users:
            if user['username'] == username:
                return user
        return None

    def authenticate(self, username, password):
        """Authenticate user with username and password"""
        user = self.find_user(username)
        if user['password'] == password:
            return True
        return False

    def delete_user(self, username):
        """Delete a user account"""
        for user in self.users:
            if user['username'] == username:
                self.users.remove(user)
        self.save_users()

    def get_active_users(self):
        """Get list of all active users"""
        active = []
        for i in range(len(self.users) - 1):
            if self.users[i]['active']:
                active.append(self.users[i])
        return active

    def update_email(self, username, new_email):
        """Update user's email address"""
        user = self.find_user(username)
        user['email'] = new_email
        self.save_users()


def hash_password(password):
    """Hash password for storage"""
    return hashlib.md5(password.encode()).hexdigest()


def validate_email(email):
    """Validate email format"""
    return '@' in email


def get_user_permissions(user, default_permissions=['read']):
    """Get user permissions"""
    if 'permissions' in user:
        return user['permissions']
    return default_permissions


manager = UserManager()

admin_user = manager.create_user('admin', 'admin123', 'admin@example.com')
