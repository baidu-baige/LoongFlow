""" Initial program for digit puzzles that remain true when viewed upside down. """

# EVOLVE-BLOCK-START
"""
An optimized solver for digit puzzles that remain true when viewed upside down.
This solution uses constraint-based filtering and early termination to reduce
the search space and improve performance, while ensuring all digits are unique.
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
    # Equation: AB + C = D * 9, where all digits A, B, C, D must be unique
    for a in VALID_DIGITS:
        if a == 0:
            continue  # Leading digit cannot be 0
        
        for b in VALID_DIGITS:
            if b == a:
                continue  # Ensure digit uniqueness
            two_digit = a * 10 + b
            
            for c in VALID_DIGITS:
                if c == a or c == b:
                    continue  # Ensure digit uniqueness
                
                # Calculate what d should be from the equation: AB + C = D * 9
                # So D = (AB + C) / 9
                left_side = two_digit + c
                if left_side % 9 != 0:
                    continue
                
                d = left_side // 9
                if d < 0 or d > 9:
                    continue
                if d == a or d == b or d == c:  # Ensure digit uniqueness
                    continue
                
                # Check upside-down equation
                sa, sb, sc, sd = str(a), str(b), str(c), str(d)
                ud_a, ud_b, ud_c, ud_d = (
                    UPSIDE_DOWN_MAP[sb], UPSIDE_DOWN_MAP[sa],  # AB becomes BA when upside down
                    UPSIDE_DOWN_MAP[sc], UPSIDE_DOWN_MAP[sd]
                )

                # Check if upside-down digits are valid and unique
                ud_digits = {ud_a, ud_b, ud_c, ud_d}
                if len(ud_digits) != 4:  # All must be unique
                    continue
                
                ud_num1 = int(ud_b + ud_a)  # BA (reversed from AB)
                ud_num2 = int(ud_c)
                ud_num3 = int(ud_d)

                if ud_num1 + ud_num2 == ud_num3 * 9:
                    solution_str = f"{a}{b} + {c} = {d} * 9"
                    solutions['puzzle1'].append(solution_str)

    # --- Puzzle 2: xx = xx - x * 4 ---
    # Equation: AB = CD - E * 4, where all digits A, B, C, D, E must be unique
    for a in VALID_DIGITS:
        if a == 0:
            continue  # Leading digit cannot be 0
        
        for b in VALID_DIGITS:
            if b == a:
                continue  # Ensure digit uniqueness
            ab = a * 10 + b
            
            for c in VALID_DIGITS:
                if c == 0:
                    continue  # Leading digit cannot be 0
                if c == a or c == b:
                    continue  # Ensure digit uniqueness
                
                for d in VALID_DIGITS:
                    if d == a or d == b or d == c:
                        continue  # Ensure digit uniqueness
                    cd = c * 10 + d
                    
                    for e in VALID_DIGITS:
                        if e == a or e == b or e == c or e == d:
                            continue  # Ensure digit uniqueness
                        
                        # Check the equation: AB = CD - E * 4
                        if ab != cd - e * 4:
                            continue
                        
                        # Check upside-down equation
                        sa, sb, sc, sd, se = str(a), str(b), str(c), str(d), str(e)
                        ud_a, ud_b, ud_c, ud_d, ud_e = (
                            UPSIDE_DOWN_MAP[sb], UPSIDE_DOWN_MAP[sa],  # AB becomes BA when upside down
                            UPSIDE_DOWN_MAP[sd], UPSIDE_DOWN_MAP[sc],  # CD becomes DC when upside down
                            UPSIDE_DOWN_MAP[se]
                        )
                        
                        # Check if upside-down digits are valid and unique
                        ud_digits = {ud_a, ud_b, ud_c, ud_d, ud_e}
                        if len(ud_digits) != 5:  # All must be unique
                            continue
                        
                        ud_num1 = int(ud_b + ud_a)  # BA (reversed from AB)
                        ud_num2 = int(ud_d + ud_c)  # DC (reversed from CD)
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
