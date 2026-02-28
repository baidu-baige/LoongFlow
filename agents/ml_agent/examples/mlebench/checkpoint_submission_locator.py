#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Locate the latest checkpoint and extract submission_file_path from best_solution.json.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any


def _checkpoint_sort_key(path: Path) -> tuple[int, int, int, float, str]:
    m = re.search(r"checkpoint-iter-(\d+)-(\d+)$", path.name)
    mtime = path.stat().st_mtime
    if m:
        return (1, int(m.group(1)), int(m.group(2)), mtime, path.name)
    return (0, 0, 0, mtime, path.name)


def find_latest_checkpoint_dir(checkpoints_root: Path) -> Path:
    if not checkpoints_root.is_dir():
        raise FileNotFoundError(f"Checkpoint root not found: {checkpoints_root}")

    checkpoint_dirs = [p for p in checkpoints_root.iterdir() if p.is_dir()]
    if not checkpoint_dirs:
        raise FileNotFoundError(f"No checkpoint directory found under: {checkpoints_root}")

    checkpoint_dirs.sort(key=_checkpoint_sort_key)
    return checkpoint_dirs[-1]


def _load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"Failed to parse JSON: {path}") from exc


def extract_submission_path(best_solution_path: Path) -> str:
    if not best_solution_path.is_file():
        raise FileNotFoundError(f"best_solution.json not found: {best_solution_path}")

    best_solution = _load_json(best_solution_path)
    evaluation = best_solution.get("evaluation", {})

    if isinstance(evaluation, str):
        try:
            evaluation = json.loads(evaluation)
        except Exception as exc:
            raise ValueError("`evaluation` is string but not valid JSON.") from exc

    if not isinstance(evaluation, dict):
        raise ValueError("`evaluation` must be a dict or JSON string.")

    artifacts = evaluation.get("artifacts", {})
    if not isinstance(artifacts, dict):
        raise ValueError("`evaluation.artifacts` is missing or invalid.")

    submission_file_path = artifacts.get("submission_file_path")
    if not isinstance(submission_file_path, str) or not submission_file_path.strip():
        raise ValueError("`evaluation.artifacts.submission_file_path` is missing or invalid.")

    return submission_file_path.strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Locate latest checkpoint and extract submission_file_path."
    )
    parser.add_argument(
        "--checkpoints-root",
        type=Path,
        required=True,
        help="Path to checkpoints root directory, e.g. output/database/checkpoints",
    )
    parser.add_argument(
        "--field",
        choices=("checkpoint_path", "best_solution_path", "submission_file_path", "json"),
        default="json",
        help="Which field to print.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        checkpoint_path = find_latest_checkpoint_dir(args.checkpoints_root)
        best_solution_path = checkpoint_path / "best_solution.json"
        submission_file_path = extract_submission_path(best_solution_path)
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    payload = {
        "checkpoint_path": str(checkpoint_path),
        "best_solution_path": str(best_solution_path),
        "submission_file_path": submission_file_path,
    }

    if args.field == "json":
        print(json.dumps(payload, ensure_ascii=False))
    else:
        print(payload[args.field])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
