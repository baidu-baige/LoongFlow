"""uncertainty_inequality problem."""

import numpy as np
import sympy
import scipy.optimize
from numpy.polynomial.hermite import Hermite
from numpy.polynomial.polynomial import Polynomial


def verify_hermite_combination(coeffs: np.ndarray) -> sympy.Expr:
    """Computes the Hermite combination for given coefficients."""
    m = len(coeffs)
    rational_coeffs = [sympy.Rational(c) for c in coeffs]
    degrees = np.arange(0, 4 * m + 4, 4)
    x = sympy.symbols("x")
    hps = [
        sympy.polys.orthopolys.hermite_poly(n=i, x=x, polys=False) for i in degrees
    ]

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
    return res_fn


# Precomputed constants for Hermite polynomials at x=0
# H_n(0) = (-1)^(n/2) * n! / (n/2)!
H0_0 = 1.0
H4_0 = 12.0
H8_0 = 1680.0
H12_0 = 665280.0


def objective_function(coeffs):
    """
    Calculates the upper bound C4 for the given coefficients.
    We want to minimize this value.
    """
    c0, c1, c2 = coeffs

    # Calculate c3 to satisfy P(0) = 0
    # Constraint equation: c0*H0(0) + c1*H4(0) + c2*H8(0) + c3*H12(0) = 0
    numerator = c0 * H0_0 + c1 * H4_0 + c2 * H8_0

    # Avoid division by zero
    c3 = -numerator / H12_0

    # Construct the Hermite series
    # The coefficients array for numpy Hermite corresponds to [H0, H1, H2, ...]
    # We populate indices 0, 4, 8, 12
    h_coeffs = np.zeros(13)
    h_coeffs[0] = c0
    h_coeffs[4] = c1
    h_coeffs[8] = c2
    h_coeffs[12] = c3

    # Create the Hermite polynomial object
    H = Hermite(h_coeffs)

    # Convert to standard polynomial basis to find roots easily
    try:
        P = H.convert(kind=Polynomial)
    except Exception:
        return 1e9  # Penalty for conversion failure

    # Find roots of the polynomial
    roots = P.roots()

    # Filter for real roots
    # Allow small imaginary part due to floating point noise
    real_mask = np.abs(roots.imag) < 1e-6
    real_roots = roots[real_mask].real

    # Filter for strictly positive roots
    # We ignore the root at 0 (and any negative roots due to symmetry/construction)
    # Use a threshold to filter out the root at 0
    pos_roots = real_roots[real_roots > 1e-4]

    if len(pos_roots) == 0:
        # If no positive roots, the function doesn't define a valid A(f) > 0
        # or implies A(f)=0 which is physically invalid for this problem
        return 1e9

    # A(f) is the largest positive root
    largest_root = np.max(pos_roots)

    # Calculate the upper bound C4
    # C4 <= A(f)^2 / (2 * pi)
    bound = (largest_root ** 2) / (2 * np.pi)

    return bound


def find_coefficients():
    """
    Find the coefficients of the problem using Differential Evolution.
    """
    # Define bounds for c0, c1, c2.
    # The search space is continuous. Large bounds prevent constraint issues.
    bounds = [(-100, 100), (-100, 100), (-100, 100)]

    # Use Differential Evolution to find global minimum
    result = scipy.optimize.differential_evolution(
        objective_function,
        bounds,
        strategy='best1bin',
        maxiter=2000,
        popsize=50,
        tol=1e-8,
        mutation=(0.5, 1.0),
        recombination=0.7,
        seed=42,  # Fixed seed for reproducibility
        polish=True
    )

    best_coefficients = result.x
    return best_coefficients

def get_upper_bound(coeffs: np.ndarray) -> float:
  """Computes the upper bound for the given Hermite combination."""
  g_exp = verify_hermite_combination(coeffs)

  x = sympy.symbols('x')
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
        (gq_fn.subs(x, approx_root_p) > 0 and gq_fn.subs(x, approx_root_m) < 0)
        or (
            gq_fn.subs(x, approx_root_p) < 0
            and gq_fn.subs(x, approx_root_m) > 0
        )
    )
    if is_sign_change:
      largest_sign_change = max(largest_sign_change, approx_root)

  return float(largest_sign_change**2) / (2 * np.pi)


if __name__ == "__main__":
    """Run the uncertainty inequality find_coefficients()"""
    np.set_printoptions(suppress=True, precision=20)
    best_coefficients = find_coefficients()
    print(best_coefficients)
    print(get_upper_bound(best_coefficients))