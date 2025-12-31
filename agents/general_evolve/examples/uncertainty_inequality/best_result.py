"""uncertainty_inequality problem."""

import numpy as np
import sympy

coefficients_evolux = np.array(
    [96.65316337499632, -3.4008547231204944, -0.026189743258153708]
)


def verify_hermite_combination(inputs: sympy.Expr):
    """verifies that the given Hermite combination exists."""
    x = sympy.symbols("x")
    # Check the value at 0 is 0.
    value_at_0 = inputs.subs(x, 0)
    assert value_at_0 == 0, f"The value at 0 is {value_at_0} != 0."
    assert sympy.limit(inputs, x, sympy.oo) > 0, "Limit at infty is not positive."


def find_hermite_combination(coeffs: np.ndarray) -> sympy.Expr:
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

    return res_fn


def get_upper_bound(coeffs: np.ndarray) -> float:
    """Computes the upper bound for the given Hermite combination."""
    g_exp = find_hermite_combination(coeffs)
    x = sympy.symbols("x")
    gq_fn = sympy.exquo(g_exp, x**2)
    rroots = sympy.real_roots(gq_fn, x)
    approx_roots = list()
    largest_sign_change = 0

    for root in rroots:
        approx_root = root.eval_rational(n=200)
        approx_root_p = approx_root + sympy.Rational(1e-198)
        approx_root_m = approx_root - sympy.Rational(1e-198)
        approx_roots.append(approx_root)
        is_sign_change = (
            gq_fn.subs(x, approx_root_p) > 0 and gq_fn.subs(x, approx_root_m) < 0
        ) or (gq_fn.subs(x, approx_root_p) < 0 and gq_fn.subs(x, approx_root_m) > 0)
        if is_sign_change:
            largest_sign_change = max(largest_sign_change, approx_root)

    return float(largest_sign_change**2) / (2 * np.pi)


# @title Verifier
if __name__ == "__main__":
    evolux_exp = find_hermite_combination(coefficients_evolux)
    verify_hermite_combination(evolux_exp)

    print(
        "LoongFlow construction is correct and gives the upper bound: C4 <=",
        get_upper_bound(coefficients_evolux),
    )
