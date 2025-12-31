# -*- coding: utf-8 -*-
"""
This file provides logger utilities.
"""

import logging
import sys

from agentsdk.logger.context import get_log_id, new_log_id, set_log_id


class TraceIdFilter(logging.Filter):
    """
    Filter class that adds a 'log_id' field to log records.
    """
    def __init__(self):
        super().__init__()

    def filter(self, record: logging.LogRecord) -> bool:
        """
        Filter function to add log_id to log records.
        """
        if hasattr(record, "log_id"):
            return True
        log_id = get_log_id()
        if log_id is None:
            log_id = new_log_id()
            set_log_id(log_id)
        record.log_id = log_id
        return True

def get_logger(name: str = None) -> logging.Logger:
    """
    Get a logger safely. 
    - If name is None, returns root logger.
    - If name is given, returns child logger that propagates to root logger.
    - Ensures no duplicate handlers are added.
    """
    root_logger = logging.getLogger()
    if not root_logger.handlers:
        # Add default console handler to root logger
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [log_id=%(log_id)s] [%(name)s] %(message)s"
        )
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.addFilter(TraceIdFilter())
        root_logger.addHandler(console_handler)
        root_logger.setLevel(logging.INFO)

    logger = logging.getLogger(name)
    logger.propagate = True
    return logger