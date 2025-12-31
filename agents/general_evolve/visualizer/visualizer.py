#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Flask web server for the evolution tree visualizer.

The server exposes REST endpoints to read checkpoint metadata and tree
structures. The front-end (pure HTML/CSS/JS) is located under `static/`.
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
from pathlib import Path
from typing import Any, Dict, List

from flask import Flask, abort, jsonify, request, send_from_directory

BASE_DIR = Path(__file__).parent.parent.parent.parent


class CheckpointService:
    """Service that loads checkpoint data and builds evolution trees."""

    def __init__(self, checkpoint_root: str):
        checkpoint_path = Path(checkpoint_root)
        if not checkpoint_path.is_absolute():
            checkpoint_path = BASE_DIR / checkpoint_path
        self.root_path = checkpoint_path.resolve()
        if not self.root_path.exists():
            raise FileNotFoundError(f"Checkpoint root not found: {self.root_path}")

    # ------------------------------------------------------------------
    # Public APIs
    # ------------------------------------------------------------------
    def list_checkpoints(self) -> List[str]:
        """Return a sorted list of checkpoint folder names."""
        checkpoints = []
        for child in self.root_path.iterdir():
            if not child.is_dir():
                continue
            if (child / "metadata.json").exists():
                checkpoints.append(child.name)

        # Sort by iteration number if available, otherwise lexicographically.
        checkpoints.sort(key=self._checkpoint_sort_key)
        return checkpoints

    def get_checkpoint_data(self, checkpoint_id: str) -> Dict[str, Any]:
        """Return tree + metadata for the requested checkpoint."""
        checkpoint_path = self._resolve_checkpoint_path(checkpoint_id)
        metadata = self._load_metadata(checkpoint_path)
        solutions = self._load_solutions(checkpoint_path)

        # Build islands data from solutions by grouping by island_id
        islands = self._build_islands_from_solutions(solutions)

        tree = self._build_tree(solutions, metadata, islands)
        score_history = self._build_score_history(solutions, metadata)

        stats = {
            "total_solutions": len(solutions),
            "total_valid_solutions": metadata.get("total_valid_solutions", 0),
            "best_score": metadata.get("feature_stats", {})
            .get("score", {})
            .get("max", 0.0),
            "last_iteration": metadata.get("last_iteration", 0),
        }

        return {
            "tree": tree,
            "metadata": metadata,
            "stats": stats,
            "islands": islands,
            "score_history": score_history,
        }

    def get_solution_diff(
        self, checkpoint_id: str, current_node_id: str, parent_node_id: str
    ) -> Dict[str, Any]:
        """Calculate diff between current and parent solution codes and merge with current code."""
        checkpoint_path = self._resolve_checkpoint_path(checkpoint_id)
        solutions = self._load_solutions(checkpoint_path)

        # Get current and parent solutions
        current_solution = solutions.get(current_node_id)
        parent_solution = solutions.get(parent_node_id)

        if not current_solution:
            raise ValueError(f"Solution '{current_node_id}' not found")
        if not parent_solution:
            raise ValueError(f"Parent solution '{parent_node_id}' not found")

        current_code = current_solution.get("solution", "")
        parent_code = parent_solution.get("solution", "")

        # If codes are identical, return early
        if current_code == parent_code:
            return {
                "diff": "",
                "has_changes": False,
                "message": "Code has no changes.",
                "merged_code": current_code,
                "lines": [],
            }

        # Split into lines for comparison (without keepends to get clean lines)
        current_lines = current_code.splitlines()
        parent_lines = parent_code.splitlines()

        # Use SequenceMatcher to find matching blocks
        matcher = difflib.SequenceMatcher(None, parent_lines, current_lines)

        # Build line-by-line diff information
        merged_lines = []
        current_line_num = 0
        parent_line_num = 0

        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "equal":
                # Unchanged lines
                for line in current_lines[j1:j2]:
                    merged_lines.append(
                        {
                            "line": line,
                            "status": "unchanged",
                            "line_number": current_line_num + 1,
                        }
                    )
                    current_line_num += 1
                    parent_line_num += 1
            elif tag == "replace":
                # Modified lines: show removed lines first, then added lines
                # Show removed lines (from parent)
                for line in parent_lines[i1:i2]:
                    merged_lines.append(
                        {
                            "line": line,
                            "status": "removed",
                            "line_number": parent_line_num + 1,
                            "old_line_number": parent_line_num + 1,
                        }
                    )
                    parent_line_num += 1
                # Show added lines (from current)
                for line in current_lines[j1:j2]:
                    merged_lines.append(
                        {
                            "line": line,
                            "status": "added",
                            "line_number": current_line_num + 1,
                        }
                    )
                    current_line_num += 1
            elif tag == "delete":
                # Removed lines
                for line in parent_lines[i1:i2]:
                    merged_lines.append(
                        {
                            "line": line,
                            "status": "removed",
                            "line_number": parent_line_num + 1,
                            "old_line_number": parent_line_num + 1,
                        }
                    )
                    parent_line_num += 1
            elif tag == "insert":
                # Added lines
                for line in current_lines[j1:j2]:
                    merged_lines.append(
                        {
                            "line": line,
                            "status": "added",
                            "line_number": current_line_num + 1,
                        }
                    )
                    current_line_num += 1

        # Generate unified diff for reference
        diff_lines = list(
            difflib.unified_diff(
                parent_lines,
                current_lines,
                fromfile=f"a/{parent_node_id}",
                tofile=f"b/{current_node_id}",
                lineterm="",
            )
        )
        diff_text = "".join(diff_lines)

        # Reconstruct merged code with diff markers
        merged_code_lines = []
        for line_info in merged_lines:
            status = line_info["status"]
            line_content = line_info["line"]
            if status == "removed":
                merged_code_lines.append(f"- {line_content}")
            elif status == "added":
                merged_code_lines.append(f"+ {line_content}")
            else:
                merged_code_lines.append(f"  {line_content}")

        merged_code = "\n".join(merged_code_lines)

        return {
            "diff": diff_text,
            "has_changes": True,
            "current_id": current_node_id,
            "parent_id": parent_node_id,
            "merged_code": merged_code,
            "lines": merged_lines,
            "current_code": current_code,
            "parent_code": parent_code,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _checkpoint_sort_key(self, name: str) -> tuple[int, str]:
        match = re.search(r"iter-(\\d+)", name)
        iteration = int(match.group(1)) if match else -1
        return (iteration, name)

    def _resolve_checkpoint_path(self, checkpoint_id: str) -> Path:
        if Path(checkpoint_id).name != checkpoint_id:
            raise ValueError("Invalid checkpoint id.")

        checkpoint_path = (self.root_path / checkpoint_id).resolve()

        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint '{checkpoint_id}' not found under {self.root_path}"
            )
        if self.root_path not in checkpoint_path.parents:
            raise ValueError("Checkpoint path escapes the configured root.")

        return checkpoint_path

    def _load_metadata(self, checkpoint_path: Path) -> Dict[str, Any]:
        metadata_path = checkpoint_path / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"metadata.json missing in {checkpoint_path}")

        with open(metadata_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _load_solutions(self, checkpoint_path: Path) -> Dict[str, Dict[str, Any]]:
        solutions_dir = checkpoint_path / "solutions"
        if not solutions_dir.exists():
            raise FileNotFoundError(f"'solutions' folder missing in {checkpoint_path}")

        solutions: Dict[str, Dict[str, Any]] = {}
        for solution_file in solutions_dir.glob("*.json"):
            with open(solution_file, "r", encoding="utf-8") as f:
                solution = json.load(f)
            solution_id = solution.get("solution_id")
            if solution_id:
                solutions[solution_id] = solution
        return solutions

    def _build_islands_from_solutions(
        self, solutions: Dict[str, Dict[str, Any]]
    ) -> List[List[str]]:
        """Build islands data by grouping solutions by island_id."""
        # Group solutions by island_id
        island_map: Dict[int, List[str]] = {}
        for solution_id, solution in solutions.items():
            island_id = solution.get("island_id")
            if island_id is not None:
                island_id = int(island_id)
                if island_id not in island_map:
                    island_map[island_id] = []
                island_map[island_id].append(solution_id)

        # Convert to list format, ensuring all island indices are present
        if not island_map:
            return []

        max_island_id = max(island_map.keys())
        islands: List[List[str]] = []
        for island_id in range(max_island_id + 1):
            islands.append(island_map.get(island_id, []))

        return islands

    def _build_tree(
        self,
        solutions: Dict[str, Dict[str, Any]],
        metadata: Dict[str, Any],
        islands: List[List[str]],
    ) -> Dict[str, Any]:
        if not solutions:
            return {"id": "empty", "name": "No Data", "children": []}

        elite_ids = set(metadata.get("elites", []))
        best_solution_id = metadata.get("best_solution_id")
        island_best_ids = set[Any](metadata.get("island_best_solution", []))
        # Build a map from solution_id to island index for quick lookup
        solution_to_island: Dict[str, int] = {}
        for island_idx, island_solutions in enumerate(islands):
            for solution_id in island_solutions:
                solution_to_island[solution_id] = island_idx

        def get_island_order(solution_id: str) -> int:
            """Get the island index for a solution, or a large number if not in any island."""
            return solution_to_island.get(solution_id, len(islands) + 1000)

        children_map: Dict[str, List[str]] = {}
        roots: List[str] = []

        for solution_id, solution in solutions.items():
            parent_id = solution.get("parent_id")
            if parent_id and parent_id in solutions:
                children_map.setdefault(parent_id, []).append(solution_id)
            else:
                roots.append(solution_id)

        def build_node(solution_id: str) -> Dict[str, Any]:
            solution = solutions[solution_id]
            node = {
                "id": solution_id,
                "name": solution_id,
                "solution_id": solution_id,
                "score": solution.get("score", 0.0) or 0.0,
                "generation": solution.get("generation", 0),
                "iteration": solution.get("iteration", 0),
                "island_id": solution.get("island_id", 0),
                "is_elite": solution_id in elite_ids,
                "is_best": solution_id == best_solution_id,
                "is_island_best": solution_id in island_best_ids,
                "parent_id": solution.get("parent_id"),
                "children": [],
                "solution": solution.get("solution"),
                "generate_plan": solution.get("generate_plan"),
                "timestamp": solution.get("timestamp"),
                "sample_weight": solution.get("sample_weight"),
                "score": solution.get("score"),
                "evaluation": solution.get("evaluation"),
                "summary": solution.get("summary"),
            }

            child_ids = children_map.get(solution_id, [])
            # Sort by island order first, then by iteration
            child_ids.sort(
                key=lambda cid: (
                    get_island_order(cid),
                    solutions[cid].get("iteration", 0),
                )
            )
            node["children"] = [build_node(child_id) for child_id in child_ids]
            return node

        # Sort roots by island order first, then by iteration
        roots.sort(
            key=lambda rid: (get_island_order(rid), solutions[rid].get("iteration", 0))
        )
        trees = [build_node(root_id) for root_id in roots]

        if len(trees) == 1:
            return trees[0]
        return {"id": "root", "name": "Root", "children": trees}

    def _build_score_history(
        self, solutions: Dict[str, Dict[str, Any]], metadata: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Collect all solution scores by iteration for charting."""
        best_solution_id = metadata.get("best_solution_id")
        elite_ids = set(metadata.get("elites", []))
        island_best_ids = set[Any](metadata.get("island_best_solution", []))

        history: List[Dict[str, Any]] = []
        for solution_id, solution in solutions.items():
            iteration = solution.get("iteration")
            score = solution.get("score")
            if iteration is None or score is None:
                continue

            history.append(
                {
                    "iteration": int(iteration),
                    "score": float(score),
                    "solution_id": solution_id,
                    "is_best": solution_id == best_solution_id,
                    "is_elite": solution_id in elite_ids,
                    "is_island_best": solution_id in island_best_ids,
                }
            )

        history.sort(key=lambda item: (item["iteration"], -item["score"]))
        return history


def create_app(checkpoint_root: str) -> Flask:
    """
    Create a Visualizer application.
    """
    app = Flask(
        __name__,
        static_folder=str(Path(__file__).parent / "static"),
        static_url_path="/static",
    )
    service = CheckpointService(checkpoint_root)

    @app.route("/", methods=["GET"])
    def index() -> Any:
        return send_from_directory(app.static_folder, "index.html")

    @app.route("/api/checkpoints", methods=["GET"])
    def list_checkpoints_endpoint() -> Any:
        # Rank checkpoints by iteration number
        checkpoints = sorted(
            service.list_checkpoints(), key=lambda x: int(x.split("-")[-1])
        )
        default_checkpoint = checkpoints[-1] if checkpoints else None
        return jsonify({"checkpoints": checkpoints, "default": default_checkpoint})

    @app.route("/api/checkpoints/<checkpoint_id>/tree", methods=["GET"])
    def checkpoint_tree_endpoint(checkpoint_id: str) -> Any:
        try:
            data = service.get_checkpoint_data(checkpoint_id)
        except FileNotFoundError:
            abort(404, description="Checkpoint not found.")
        except ValueError as exc:
            abort(400, description=str(exc))

        return jsonify({"checkpoint": checkpoint_id, **data})

    @app.route("/api/checkpoints/<checkpoint_id>/diff", methods=["GET"])
    def solution_diff_endpoint(checkpoint_id: str) -> Any:
        """Calculate diff between current and parent solution."""
        current_node_id = request.args.get("current_node_id")
        parent_node_id = request.args.get("parent_node_id")

        if not current_node_id:
            abort(400, description="Missing required parameter: current_node_id")
        if not parent_node_id:
            abort(400, description="Missing required parameter: parent_node_id")

        try:
            diff_data = service.get_solution_diff(
                checkpoint_id, current_node_id, parent_node_id
            )
            return jsonify(diff_data)
        except FileNotFoundError:
            abort(404, description="Checkpoint not found.")
        except ValueError as exc:
            abort(400, description=str(exc))

    @app.route("/health", methods=["GET"])
    def health_check() -> Any:
        return jsonify({"status": "ok"})

    return app


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Run the evolution tree visualizer web server."
    )
    parser.add_argument(
        "--checkpoint-path",
        default="output/database/checkpoints",
        help="Directory containing checkpoint folders.",
    )
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind.")
    parser.add_argument("--port", type=int, default=5000, help="Port to listen on.")
    parser.add_argument(
        "--debug", action="store_true", help="Enable Flask debug/reload mode."
    )
    return parser.parse_args()


def main() -> None:
    """
    Main function to run the visualizer.
    """
    args = parse_args()
    app = create_app(args.checkpoint_path)

    if args.debug:
        app.run(host=args.host, port=args.port, debug=True)
    else:
        try:
            import uvicorn

            try:
                from asgiref.wsgi import WsgiToAsgi

                asgi_app = WsgiToAsgi(app)
            except ImportError:
                from uvicorn.middleware.wsgi import WSGIMiddleware

                asgi_app = WSGIMiddleware(app)

            print(f" * Starting uvicorn server on http://{args.host}:{args.port}")
            uvicorn.run(asgi_app, host=args.host, port=args.port, log_level="info")
        except ImportError:
            try:
                from gevent.pywsgi import WSGIServer

                print(f" * Starting gevent server on http://{args.host}:{args.port}")
                http_server = WSGIServer((args.host, args.port), app)
                http_server.serve_forever()
            except ImportError:
                print(
                    " * WARNING: Production server (uvicorn/gevent) not found. Using Flask dev server."
                )
                app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
