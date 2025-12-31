""" Best solution found by evolution for math_flip task. """

# EVOLVE-BLOCK-START
"""
An optimized solver for digit puzzles that remain true when viewed upside down.
This solution uses a more efficient approach by leveraging equation constraints
to reduce the search space and avoid unnecessary computations.
"""


def solve_puzzles():
    """
    Finds solutions to the two digit puzzles based on the given rules.

    Returns:
        dict: A dictionary containing the solutions for each puzzle.
              e.g., {'puzzle1': ['solution_string'], 'puzzle2': ['solution_string']}
    """
    # Mapping of digits to their upside-down counterparts.
    UPSIDE_DOWN_MAP = {
        '0': '0', '1': '1', '2': '2', '3': '3', '4': '4',
        '5': '5', '6': '6', '7': '7', '8': '8', '9': '9'
    }
    VALID_DIGITS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    solutions = {'puzzle1': [], 'puzzle2': []}

    # --- Puzzle 1: xx + x = x * 9 ---
    # Looking for equation: AB + C = D * 9
    for a in VALID_DIGITS:
        for b in VALID_DIGITS:
            if a == 0:  # xx must be a two-digit number
                continue
            ab = a * 10 + b
            for c in VALID_DIGITS:
                for d in VALID_DIGITS:
                    if ab + c == d * 9:
                        digits = {a, b, c, d}
                        if len(digits) == 4:  # All digits must be unique
                            # Check upside-down equation
                            sa, sb, sc, sd = str(a), str(b), str(c), str(d)
                            ud_a = UPSIDE_DOWN_MAP[sb]  # Note: ab flips to ud_b + ud_a
                            ud_b = UPSIDE_DOWN_MAP[sa]
                            ud_c = UPSIDE_DOWN_MAP[sc]
                            ud_d = UPSIDE_DOWN_MAP[sd]
                            
                            ud_num1 = int(ud_b + ud_a)
                            ud_num2 = int(ud_c)
                            ud_num3 = int(ud_d)
                            
                            if ud_num1 + ud_num2 == ud_num3 * 9:
                                solution_str = f"{a}{b} + {c} = {d} * 9"
                                solutions['puzzle1'].append(solution_str)

    # --- Puzzle 2: xx = xx - x * 4 ---
    # Looking for equation: AB = CD - E * 4
    for a in VALID_DIGITS:
        for b in VALID_DIGITS:
            if a == 0:  # xx must be a two-digit number
                continue
            ab = a * 10 + b
            for c in VALID_DIGITS:
                for d in VALID_DIGITS:
                    if c == 0:  # xx must be a two-digit number
                        continue
                    cd = c * 10 + d
                    for e in VALID_DIGITS:
                        if ab == cd - e * 4:
                            digits = {a, b, c, d, e}
                            if len(digits) == 5:  # All digits must be unique
                                # Check upside-down equation
                                sa, sb, sc, sd, se = str(a), str(b), str(c), str(d), str(e)
                                ud_a = UPSIDE_DOWN_MAP[sb]  # ab flips to ud_b + ud_a
                                ud_b = UPSIDE_DOWN_MAP[sa]
                                ud_c = UPSIDE_DOWN_MAP[sd]  # cd flips to ud_d + ud_c
                                ud_d = UPSIDE_DOWN_MAP[sc]
                                ud_e = UPSIDE_DOWN_MAP[se]
                                
                                ud_num1 = int(ud_b + ud_a)
                                ud_num2 = int(ud_d + ud_c)
                                ud_num3 = int(ud_e)
                                
                                if ud_num1 == ud_num2 - ud_num3 * 4:
                                    solution_str = f"{a}{b} = {c}{d} - {e} * 4"
                                    solutions['puzzle2'].append(solution_str)

    return solutions

# EVOLVE-BLOCK-END


# This part remains fixed (not evolved)
def run():
    """
    Main entry point to run the puzzle solver.
    This function calls the core solver logic and returns its result.
    """
    # solve_puzzles must return example: {'puzzle1': ['solution_string'], 'puzzle2': ['solution_string']}
    solutions = solve_puzzles()
    return solutions


if __name__ == "__main__":
    # When the script is run directly, execute the solver and print the results.
    results = run()
    print(results)