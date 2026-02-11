#!/usr/bin/env python3
"""
Test script for General Agent Visualizer.
Verifies that the visualizer can correctly parse and display task data.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from agents.general_agent.visualizer.visualizer import GeneralAgentService


def test_visualizer():
    """Test the visualizer with sample data."""
    print("=" * 60)
    print("Testing General Agent Visualizer")
    print("=" * 60)

    # Try to find some workspace directories
    base_dir = Path(__file__).parent.parent.parent.parent
    possible_workspaces = [
        base_dir / "output-todo-list",
        base_dir / "output-file-processor",
        base_dir / "output-bug-hunter",
        base_dir / "output-circle-packing",
    ]

    found_workspaces = [str(ws) for ws in possible_workspaces if ws.exists()]

    if not found_workspaces:
        print("\n⚠️  No workspace directories found!")
        print("\nTo test the visualizer, first run a task:")
        print("  ./run_general.sh 01_todo_list\n")
        print("Then run this test again:")
        print("  python agents/general_agent/visualizer/test_visualizer.py\n")
        return False

    print(f"\n✓ Found {len(found_workspaces)} workspace(s):")
    for ws in found_workspaces:
        print(f"  - {ws}")

    # Initialize service
    print("\n→ Initializing visualizer service...")
    try:
        service = GeneralAgentService(found_workspaces)
        print("✓ Service initialized successfully")
    except Exception as e:
        print(f"✗ Failed to initialize service: {e}")
        return False

    # Test list_tasks
    print("\n→ Testing task listing...")
    try:
        tasks = service.list_tasks()
        print(f"✓ Found {len(tasks)} task(s)")

        if tasks:
            for task in tasks[:3]:  # Show first 3
                print(f"  - {task['task_name']}: "
                      f"{task['num_iterations']} iterations, "
                      f"score: {task['latest_score']}")
        else:
            print("  (no tasks with iterations yet)")
    except Exception as e:
        print(f"✗ Failed to list tasks: {e}")
        return False

    # Test task details if we have tasks
    if tasks:
        print("\n→ Testing task details...")
        try:
            task_id = tasks[0]['task_id']
            details = service.get_task_details(task_id)
            print(f"✓ Retrieved details for: {task_id}")
            print(f"  - Iterations: {details['current_iteration']}")
            print(f"  - Score history: {len(details['score_history'])} points")
            if details['best_score']:
                print(f"  - Best score: {details['best_score']['score']:.3f} "
                      f"at iteration {details['best_score']['iteration']}")
        except Exception as e:
            print(f"✗ Failed to get task details: {e}")
            return False

        # Test iteration details if we have iterations
        if details['iterations']:
            print("\n→ Testing iteration details...")
            try:
                iteration_num = details['iterations'][0]['iteration']
                iter_details = service.get_iteration_details(task_id, iteration_num)
                print(f"✓ Retrieved iteration {iteration_num} details")
                print(f"  - Has plan: {iter_details['plan'] is not None}")
                print(f"  - Files: {len(iter_details['files'])}")
                if iter_details['files']:
                    print(f"    - {', '.join([f['name'] for f in iter_details['files'][:3]])}")
                print(f"  - Score: {iter_details['evaluation']['score']}")
                print(f"  - Has summary: {iter_details['summary'] is not None}")
            except Exception as e:
                print(f"✗ Failed to get iteration details: {e}")
                return False

            # Test file content
            if iter_details['files']:
                print("\n→ Testing file content retrieval...")
                try:
                    file_path = iter_details['files'][0]['path']
                    file_data = service.get_file_content(task_id, iteration_num, file_path)
                    print(f"✓ Retrieved file: {file_data['filename']}")
                    print(f"  - Language: {file_data['language']}")
                    print(f"  - Size: {len(file_data['content'])} characters")
                except Exception as e:
                    print(f"✗ Failed to get file content: {e}")
                    return False

    print("\n" + "=" * 60)
    print("✓ All tests passed!")
    print("=" * 60)
    print("\nYou can now start the visualizer:")
    print(f"  python agents/general_agent/visualizer/visualizer.py \\")
    print(f"      --workspace {found_workspaces[0]} \\")
    print(f"      --port 8080")
    print("\nThen open: http://localhost:8080")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success = test_visualizer()
    sys.exit(0 if success else 1)
