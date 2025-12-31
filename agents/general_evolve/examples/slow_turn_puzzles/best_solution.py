"""Best solution of slow turn puzzle"""

import math
import sys
import time
from collections import deque


def is_valid_move(grid, visited, row, col):
    """Check if a move is valid"""
    if not (0 <= row < len(grid) and 0 <= col < len(grid[0])):
        return False
    if visited[row][col]:
        return False
    if grid[row][col] == "R":
        return False
    return True


def calculate_turn_angle(prev_dir, current_dir):
    """Calculate if the turn angle is valid (0 or 45 degrees)"""
    if prev_dir == current_dir:
        return True

    # Calculate dot product
    dot = prev_dir[0] * current_dir[0] + prev_dir[1] * current_dir[1]

    # For 45 degree turns, dot product should be positive
    # and satisfy the condition from validation function
    if dot <= 0:
        return False

    # Check if the angle is 45 degrees
    norm_prev = math.sqrt(prev_dir[0] ** 2 + prev_dir[1] ** 2)
    norm_curr = math.sqrt(current_dir[0] ** 2 + current_dir[1] ** 2)
    cos_angle = dot / (norm_prev * norm_curr)

    # Allow 0 and 45 degree turns (cos(45°) ≈ 0.7071)
    return cos_angle >= 0.7071


# EVOLVE-BLOCK-START
def find_loop_bfs(grid, start):
    """Use BFS with pruning to find valid loop"""
    rows, cols = len(grid), len(grid[0])
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1)]

    # Track green squares that need to be visited
    green_squares = [
        (i, j) for i in range(rows) for j in range(cols) if grid[i][j] == "G"
    ]
    required_greens = len(green_squares)

    # Use BFS with state: (position, visited_mask, path, last_direction)
    queue = deque()

    # Initialize with all possible directions from start
    for direction in directions:
        next_row, next_col = start[0] + direction[0], start[1] + direction[1]
        if is_valid_move(
            grid, [[False] * cols for _ in range(rows)], next_row, next_col
        ):
            new_visited = [[False] * cols for _ in range(rows)]
            new_visited[start[0]][start[1]] = True
            new_visited[next_row][next_col] = True

            # Track green squares visited
            greens_visited = 0
            if grid[start[0]][start[1]] == "G":
                greens_visited += 1
            if grid[next_row][next_col] == "G":
                greens_visited += 1

            path = [start, (next_row, next_col)]
            queue.append(
                (next_row, next_col, new_visited, path, direction, greens_visited)
            )

    while queue:
        row, col, visited, path, last_direction, greens_visited = queue.popleft()

        # Check if we can return to start with valid turn
        if (row, col) == start and len(path) > 2:
            # Check if we've visited all green squares
            if greens_visited == required_greens:
                # Check final turn is valid
                prev_dir = (row - path[-2][0], col - path[-2][1])
                start_dir = (path[1][0] - start[0], path[1][1] - start[1])
                if calculate_turn_angle(prev_dir, start_dir):
                    return path

        # Try all directions
        for direction in directions:
            next_row, next_col = row + direction[0], col + direction[1]

            # Skip if trying to return to start from the wrong position
            if (next_row, next_col) == start and len(path) <= 2:
                continue

            # Check if we can return to start (only if it's the next move)
            if (next_row, next_col) == start and len(path) > 2:
                # Check turn angle for returning to start
                if calculate_turn_angle(last_direction, direction):
                    # Check if we've visited all green squares
                    if greens_visited == required_greens:
                        final_path = path + [start]
                        # Verify the turn from final move to first move is valid
                        final_to_start_dir = (start[0] - row, start[1] - col)
                        start_to_next_dir = (
                            path[1][0] - start[0],
                            path[1][1] - start[1],
                        )
                        if calculate_turn_angle(final_to_start_dir, start_to_next_dir):
                            return final_path
                continue

            if is_valid_move(grid, visited, next_row, next_col):
                # Check turn angle
                if not calculate_turn_angle(last_direction, direction):
                    continue

                # Create new state
                new_visited = [row[:] for row in visited]
                new_visited[next_row][next_col] = True

                new_greens = greens_visited
                if grid[next_row][next_col] == "G":
                    new_greens += 1

                # Prune if we can't reach all green squares
                remaining_moves = len(grid) * len(grid[0]) - len(path) - 1
                if new_greens + remaining_moves < required_greens:
                    continue

                new_path = path + [(next_row, next_col)]
                queue.append(
                    (next_row, next_col, new_visited, new_path, direction, new_greens)
                )

    return None


# EVOLVE-BLOCK-END


def solve_puzzle(grid):
    """Solve the puzzle"""
    start_time = time.time()

    # Find all green squares to determine starting points
    green_squares = [
        (i, j)
        for i in range(len(grid))
        for j in range(len(grid[0]))
        if grid[i][j] == "G"
    ]

    if not green_squares:
        return None, 0

    # Try starting from each green square
    for start in green_squares:
        result = find_loop_bfs(grid, start)
        if result is not None:
            end_time = time.time()
            return result, end_time - start_time

    return None, 0


if __name__ == "__main__":
    grid = [
        ["", "", "G", "", "", ""],
        ["R", "", "", "G", "", ""],
        ["", "", "", "", "", ""],
        ["", "", "", "", "", ""],
        ["", "", "", "", "", ""],
        ["", "R", "", "", "", ""],
    ]
    solution, time_taken = solve_puzzle(grid)
    if solution:
        print(f"Find the solution: {solution}, time_taken: {time_taken}")
        sys.exit(0)
    else:
        print("No valid path found.")
        sys.exit(1)
