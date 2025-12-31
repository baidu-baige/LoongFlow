# -*- coding: utf-8 -*-
"""
This file define solution utils
"""


def simplify_solution(func, fields=None):
    """simply solution result"""

    def wrapper(*args, **kwargs):
        """
        wrapper function
        """
        data = func(*args, **kwargs)
        if not data:
            return []

        if not fields:
            # keep default fields
            return [{k: v for k, v in item.items()
                     if k in ["solution_id", "generate_plan", "parent_id",
                              "island_id", "score", "evaluation", "summary"]
                     and v is not None}
                    for item in data]
        else:
            # keep specified fields
            return [{k: item.get(k) for k in fields} for item in data]

    return wrapper
