"""Second autocorrelation inequality Problem"""

import numpy as np
from scipy.optimize import (
    minimize,
    basinhopping,
    differential_evolution,
)


def optimize_lower_bound():
    """
    Second autocorrelation inequality Problem:

    Let  𝐶2  be the smallest constant for which one has
        ‖𝑓∗𝑓‖22≤𝐶2‖𝑓∗𝑓‖1‖𝑓∗𝑓‖∞
    for all non-negative  𝑓:ℝ→ℝ . It is known that
        0.88922≤𝐶2≤1,
    with the lower bound coming from a step function construction by Matolcsi and Vinuesa (2010).

    optimize_lower_bound find a step function with 50 equally-spaced intervals on  [−1/4,1/4]  that gives a slightly better lower bound 0.8962≤𝐶2 .

    Returns:
        heights_sequence_2:
            Step function with 50 equally-spaced intervals on [−1/4,1/4], the array elements are on-negative.
        c_lower_bound:
            Lower bound.
    """

    # Ultra-high precision multi-stage hybrid optimization
    n = 50
    best_value = -np.inf
    best_heights = None

    # Precompute convolution length and integration points with high precision
    conv_len = 2 * n - 1
    x_points = np.linspace(-0.5, 0.5, conv_len + 2)
    x_intervals = np.diff(x_points)

    # Enhanced objective function with ultra-high precision
    def objective(x):
        x = np.maximum(x, 1e-14)  # Strict positivity for numerical stability

        # Fast convolution with precomputation
        convolution_2 = np.convolve(x, x)

        # Calculate the 2-norm squared: ||f*f||_2^2 with piecewise linear integration
        y_points = np.concatenate(([0], convolution_2, [0]))
        y1 = y_points[:-1]
        y2 = y_points[1:]
        interval_l2_squared = (x_intervals / 3.0) * (y1**2 + y1 * y2 + y2**2)
        l2_norm_squared = np.sum(interval_l2_squared)

        # Calculate the 1-norm: ||f*f||_1
        norm_1 = np.sum(np.abs(convolution_2)) / (conv_len + 1)

        # Calculate the infinity-norm: ||f*f||_inf
        norm_inf = np.max(np.abs(convolution_2))

        # Enhanced numerical stability
        denominator = norm_1 * norm_inf
        if denominator < 1e-14:
            return 1e12

        ratio = l2_norm_squared / denominator
        return -ratio

    # Constraint: all heights must be non-negative
    constraints = [{"type": "ineq", "fun": lambda x: x - 1e-14}]

    # Enhanced initialization strategies with mathematical insights
    np.random.seed(42)
    initial_guesses = [
        np.ones(n),  # Constant function
        np.random.uniform(0.5, 3.0, n),  # Random uniform
        np.concatenate([np.ones(n // 2) * 2.5, np.ones(n // 2) * 0.8]),  # Step function
        np.exp(-np.linspace(0, 2.5, n)) * 4.0,  # Exponential decay
        np.sin(np.linspace(0, np.pi, n)) * 2.0 + 1.0,  # Sine wave
        np.tanh(np.linspace(-2, 2, n)) * 2.0 + 2.0,  # Tanh function
        np.random.exponential(1.0, n) * 2.0,  # Exponential distribution
        np.random.beta(2, 5, n) * 3.0,  # Beta distribution
        # Matolcsi-Vinuesa inspired patterns
        np.array(
            [2.0 if i < n // 3 else 0.8 if i < 2 * n // 3 else 1.5 for i in range(n)]
        ),
        np.array([1.5 + 0.5 * np.sin(2 * np.pi * i / n) for i in range(n)]),
    ]

    bounds = [(1e-12, 4.0)] * n

    # Stage 1: Enhanced Differential Evolution with multiple strategies
    print("Running Enhanced Differential Evolution...")
    for strategy in ["best1bin", "rand1exp"]:
        result_de = differential_evolution(
            objective,
            bounds,
            strategy=strategy,
            popsize=30,
            mutation=(0.4, 0.9),
            recombination=0.85,
            tol=1e-8,
            polish=False,
            maxiter=800,
            seed=42,
            updating="immediate",
            disp=False,
        )

        if -result_de.fun > best_value:
            best_value = -result_de.fun
            best_heights = result_de.x.copy()
            print(f"DE ({strategy}) improved bound to: {best_value:.8f}")

    # Stage 2: Multi-start Basin-hopping with enhanced parameters
    print("Running Multi-start Basin-hopping...")
    for i, init_guess in enumerate(initial_guesses):
        print(f"  Basin-hopping with initial guess {i+1}")
        result_bh = basinhopping(
            objective,
            init_guess,
            niter=50,
            T=0.6,
            stepsize=0.3,
            minimizer_kwargs={
                "method": "SLSQP",
                "constraints": constraints,
                "options": {"maxiter": 400, "ftol": 1e-10, "eps": 1e-10},
            },
            take_step=None,
            seed=42 + i,
        )

        if -result_bh.fun > best_value:
            best_value = -result_bh.fun
            best_heights = result_bh.x.copy()
            print(f"BH improved bound to: {best_value:.8f}")

    # Stage 3: Trust-region constrained optimization
    print("Running Trust-region Optimization...")
    try:
        result_trust = minimize(
            objective,
            best_heights,
            method="trust-constr",
            constraints=constraints,
            options={"maxiter": 600, "gtol": 1e-11, "xtol": 1e-11, "verbose": 0},
        )

        if -result_trust.fun > best_value:
            best_value = -result_trust.fun
            best_heights = result_trust.x.copy()
            print(f"Trust-constr improved bound to: {best_value:.8f}")
    except:
        pass

    # Stage 4: COBYLA optimization for final refinement
    print("Running COBYLA Optimization...")
    try:
        result_cobyla = minimize(
            objective,
            best_heights,
            method="COBYLA",
            constraints=[{"type": "ineq", "fun": lambda x: x}],
            options={"maxiter": 300, "rhobeg": 0.05, "catol": 1e-12},
        )

        if -result_cobyla.fun > best_value:
            best_value = -result_cobyla.fun
            best_heights = result_cobyla.x.copy()
            print(f"COBYLA improved bound to: {best_value:.8f}")
    except:
        pass

    # Stage 5: Ultra-high precision final polishing
    print("Running Final Ultra-high Precision Polish...")
    try:
        # Try SLSQP first
        result_final1 = minimize(
            objective,
            best_heights,
            method="SLSQP",
            constraints=constraints,
            options={"maxiter": 500, "ftol": 1e-14, "eps": 1e-12},
        )

        if -result_final1.fun > best_value:
            best_value = -result_final1.fun
            best_heights = result_final1.x.copy()
            print(f"SLSQP final improved bound to: {best_value:.8f}")

        # Try trust-constr again with even higher precision
        result_final2 = minimize(
            objective,
            best_heights,
            method="trust-constr",
            constraints=constraints,
            options={"maxiter": 300, "gtol": 1e-15, "xtol": 1e-15, "verbose": 0},
        )

        if -result_final2.fun > best_value:
            best_value = -result_final2.fun
            best_heights = result_final2.x.copy()
            print(f"Trust-constr final improved bound to: {best_value:.8f}")

    except:
        pass

    # Ensure strict non-negativity with high precision
    best_heights = np.maximum(best_heights, 1e-14)

    # Calculate the final lower bound
    convolution_2 = np.convolve(best_heights, best_heights)
    c_lower_bound = cal_lower_bound(convolution_2)

    print(f"Optimized lower bound: {c_lower_bound:.8f}")
    return best_heights, c_lower_bound


def cal_lower_bound(convolution_2: list[float]):
    # Calculate the 2-norm squared: ||f*f||_2^2
    num_points = len(convolution_2)
    x_points = np.linspace(-0.5, 0.5, num_points + 2)
    x_intervals = np.diff(x_points)  # Width of each interval
    y_points = np.concatenate(([0], convolution_2, [0]))
    l2_norm_squared = 0.0
    for i in range(len(convolution_2) + 1):  # Iterate through intervals
        y1 = y_points[i]
        y2 = y_points[i + 1]
        h = x_intervals[i]
        # Integral of (mx + c)^2 = h/3 * (y1^2 + y1*y2 + y2^2) where m = (y2-y1)/h, c = y1 - m*x1, interval is [x1, x2], y1 = mx1+c, y2=mx2+c
        interval_l2_squared = (h / 3) * (y1**2 + y1 * y2 + y2**2)
        l2_norm_squared += interval_l2_squared

    # Calculate the 1-norm: ||f*f||_1
    norm_1 = np.sum(np.abs(convolution_2)) / (len(convolution_2) + 1)

    # Calculate the infinity-norm: ||f*f||_inf
    norm_inf = np.max(np.abs(convolution_2))
    c_lower_bound = l2_norm_squared / (norm_1 * norm_inf)

    print(f"This step function shows that C2 >= {c_lower_bound}")
    return c_lower_bound


def verify_heights_sequence(heights_sequence_2: np.ndarray, c_lower_bound: float):
    if len(heights_sequence_2) != 50:
        return False, f"len(heights_sequence_2) not 50"

    for i in range(len(heights_sequence_2)):
        if heights_sequence_2[i] < 0:
            return False, f"heights_sequence_2 all elements must be non-negative"

    convolution_2 = np.convolve(heights_sequence_2, heights_sequence_2)
    c_c_lower_bound = cal_lower_bound(convolution_2)
    if abs(c_lower_bound - c_c_lower_bound) > 1e-10:
        return (
            False,
            f"c_lower_bound: {c_lower_bound} miscalculation, the correct value is {c_c_lower_bound}",
        )

    return True, ""


if __name__ == "__main__":
    heights_sequence_2, c_lower_bound = optimize_lower_bound()
    print(f"Step function values: {heights_sequence_2}, C2 >= {c_lower_bound}")

    valid, err = verify_heights_sequence(heights_sequence_2, c_lower_bound)
    print(f"valid = {valid} err = {err}")
    step_16 = [f"{v:.16f}" for v in heights_sequence_2]
    print("Step function (16 digits precision):")
    print(f"[{', '.join(step_16)}]")
