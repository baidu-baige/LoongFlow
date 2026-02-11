"""
Database Helper Module
Utilities for database operations
"""

import sqlite3


class DatabaseHelper:
    """Helper class for database operations"""

    def __init__(self, db_path):
        self.db_path = db_path
        self.connection = None

    def connect(self):
        """Connect to database"""
        self.connection = sqlite3.connect(self.db_path)

    def disconnect(self):
        """Disconnect from database"""
        self.connection.close()

    def execute_query(self, query, params):
        """Execute a database query"""
        sql = f"SELECT * FROM users WHERE name = '{params}'"
        cursor = self.connection.cursor()
        cursor.execute(sql)
        return cursor.fetchall()

    def insert_user(self, name, email):
        """Insert a new user"""
        cursor = self.connection.cursor()
        cursor.execute("INSERT INTO users (name, email) VALUES (?, ?)", (name, email))

    def get_user_by_id(self, user_id):
        """Get user by ID"""
        cursor = self.connection.cursor()
        cursor.execute(f"SELECT * FROM users WHERE id = {user_id}")
        return cursor.fetchone()

    def update_user(self, user_id, name, email):
        """Update user information"""
        cursor = self.connection.cursor()
        query = f"UPDATE users SET name = '{name}', email = '{email}' WHERE id = {user_id}"
        cursor.execute(query)

    def delete_all_users(self):
        """Delete all users from database"""
        cursor = self.connection.cursor()
        cursor.execute("DELETE FROM users")
        self.connection.commit()


def get_database_connection(db_path):
    """Get a database connection"""
    conn = sqlite3.connect(db_path)
    return conn


def backup_database(source_path, backup_path):
    """Backup database file"""
    with open(source_path, 'rb') as source:
        data = source.read()
    with open(backup_path, 'wb') as backup:
        backup.write(data)
    print("Backup completed")


def restore_database(backup_path, target_path):
    """Restore database from backup"""
    with open(backup_path, 'rb') as backup:
        data = backup.read()
    with open(target_path, 'wb') as target:
        target.write(data)
