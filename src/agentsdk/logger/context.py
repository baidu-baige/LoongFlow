# -*- coding: utf-8 -*-
"""
This file provides logger context.
"""

import contextvars
import uuid

_log_id_var = contextvars.ContextVar("log_id", default=None)


def new_log_id() -> str:
    """
    Generate a new log_id.
    """
    return uuid.uuid4().hex[:12]


def set_log_id(log_id: str):
    """
    Set the log_id.
    """
    _log_id_var.set(log_id)


def get_log_id() -> str | None:
    """
    Get the log_id.
    """
    return _log_id_var.get()


def clear_log_id():
    """
    Clear the log_id.
    """
    _log_id_var.set(None)
