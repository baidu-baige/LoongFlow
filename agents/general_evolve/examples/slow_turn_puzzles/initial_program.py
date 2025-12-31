"""slow turn puzzle problem"""

# EVOLVE-BLOCK-START

import sys
import time


def is_valid_move(grid, path, row, col):
    """Check if a move is valid"""
    if not (0 <= row < len(grid) and 0 <= col < len(grid[0])):
        return False
    if (row, col) in path:
        return False
    if grid[row][col] == "R":
        return False
    return True


def find_loop(grid, current, path, directions):
    """Find a loop"""
    if len(path) > 1 and current == path[0]:
        if all(grid[r][c] == "G" for (r, c) in path):
            return path[:]
        return None

    for direction in directions:
        next_row, next_col = current[0] + direction[0], current[1] + direction[1]
        if is_valid_move(grid, path, next_row, next_col):
            if len(path) >= 2:
                prev_direction = (path[-1][0] - path[-2][0], path[-1][1] - path[-2][1])
                current_direction = (next_row - current[0], next_col - current[1])
                if prev_direction != current_direction:
                    continue

            path.append((next_row, next_col))
            result = find_loop(grid, (next_row, next_col), path, directions)
            if result is not None:
                return result
            path.pop()
    return None


def solve_puzzle(grid):
    """Solve the puzzle"""
    start_time = time.time()
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1)]
    start_points = [
        (i, j)
        for i in range(len(grid))
        for j in range(len(grid[0]))
        if grid[i][j] == "G"
    ]

    for start in start_points:
        result = find_loop(grid, start, [start], directions)
        if result is not None:
            end_time = time.time()
            return result, end_time - start_time

    return None, 0


# EVOLVE-BLOCK-END

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
