"""uncertainty_inequality problem."""

import numpy as np
import sympy


def verify_hermite_combination(coeffs: np.ndarray) -> sympy.Expr:
    """Computes the Hermite combination for given coefficients."""
    m = len(coeffs)
    rational_coeffs = [sympy.Rational(c) for c in coeffs]
    degrees = np.arange(0, 4 * m + 4, 4)
    x = sympy.symbols("x")
    hps = [sympy.polys.orthopolys.hermite_poly(n=i, x=x, polys=False) for i in degrees]

    # All but the last coefficient.
    partial_fn: sympy.Expr = sum(
        rational_coeffs[i] * hps[i] for i in range(len(rational_coeffs))
    )

    # Impose the condition that the root at 0 should be 0.
    a = hps[-1].subs(x, 0)
    b = -partial_fn.subs(x, 0)
    last_coeff = b / a
    rational_coeffs.append(sympy.Rational(last_coeff))

    res_fn = sum(rational_coeffs[i] * hps[i] for i in range(len(rational_coeffs)))

    if sympy.limit(res_fn, x, sympy.oo) < 0:
        res_fn = -res_fn

    x = sympy.symbols("x")
    # Check the value at 0 is 0.
    value_at_0 = res_fn.subs(x, 0)
    assert value_at_0 == 0, f"The value at 0 is {value_at_0} != 0."
    assert sympy.limit(res_fn, x, sympy.oo) > 0, "Limit at infty is not positive."


def search_coefficients():
    """Find the coefficients of the problem."""
    best_coefficients = np.array([1, 2, 3])
    return best_coefficients


def find_coefficients():
    """
    Run the search for the best coefficients.
    """
    best_coefficients = search_coefficients()
    return best_coefficients


if __name__ == "__main__":
    """Run the uncertainty inequality find_coefficients()"""
    best_coefficients = find_coefficients()
    print(f"Best coefficients: {best_coefficients}")
    verify_hermite_combination(best_coefficients)
