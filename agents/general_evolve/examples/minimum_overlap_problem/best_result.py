""" Best result """

import numpy as np
import matplotlib.pyplot as plt


best_sequence = np.array([
    4.5724419252263385e-23, 1.0190477566810736e-22, 0.0, 2.4604835967296447e-17,
    8.428039840368923e-23, 3.623852508868815e-23, 2.5282502211575487e-17,
    2.2609291165195302e-17, 1.3939670140298841e-16, 9.290331371115018e-23,
    1.1064806886831518e-17, 1.6412857307576587e-18, 0.0, 2.0422554971891317e-17,
    4.024311981892372e-18, 0.0, 8.315228158382997e-23, 0.0, 1.2720356319733453e-17,
    7.130307862181558e-23, 6.650666004964534e-24, 5.121305537577793e-17, 5.81016717020042e-17,
    0.0, 1.0665780682641064e-22, 9.196215159440437e-24, 1.7001367097852453e-18,
    8.240560831365658e-23, 0.0, 3.306338892674101e-18, 0.030081011725047607, 0.03967696930499629,
    0.19712119283148324, 0.4038213984153838, 0.5207296895794997, 0.641156909366863, 0.7016802723055905,
    0.7599905087860023, 0.7174555153294624, 0.7764669237628831, 0.8724724485390135, 0.8693025240084815,
    0.6819210912179892, 0.5648576660789613, 0.6471556900442723, 0.9022510917957125, 0.9999999999999999,
    0.9999999999999999, 0.914111663568268, 0.44477672521387573, 0.10011176901426583, 6.84801704585137e-17,
    0.0028959269622808263, 0.009182146252324257, 0.0, 2.481991663062528e-17, 5.307115640013431e-23,
    2.526037485289834e-23, 1.3693189753182863e-16, 0.16228578950445638, 0.5129470717002999,
    0.6909361401020192, 0.7101562201993928, 0.801172830613563, 0.8347464315613691, 0.9309692550302865,
    0.7358491359229524, 0.5698175069611034, 0.45749195334956794, 0.4312892812820514, 0.43631676871067426,
    0.40283525492352085, 0.44325044580022493, 0.4337914193756686, 0.49674648031370927, 0.5946682815452423,
    0.7136853699153279, 0.8108326101293181, 0.7037597186474325, 0.7515894291976858, 0.7838571734696193,
    0.7573992576720319, 0.6158726509351785, 0.5320559383553518, 0.512318305169704, 0.4324804375720788,
    0.5101748366334481, 0.46645307239595113, 0.5007612433177652, 0.36866551347819876, 0.40573648683785124,
    0.37576357828235846, 0.5012756651385601, 0.5944730081989945, 0.5353363323439104, 0.5066491579770145,
    0.5584793054787592, 0.713702839721193, 0.8508422155378677, 0.8749303768919316, 0.8389213442525211,
    0.7970369544118324, 0.7859781781516182, 0.6702677169657378, 0.5843290355688149, 0.6748588933077951,
    0.7411600435232362, 0.7610035329655964, 0.8025506026719663, 0.8356372972408176, 0.8835424836688123,
    0.847490621086031, 0.7083750798131428, 0.6979075206212586, 0.6472581915247778, 0.6683736094686117,
    0.7337929679815439, 0.7235192319273117, 0.6866208937336892, 0.8134323565029921, 0.9727085986734294,
    0.9724798797726575, 0.8959424189442907, 0.9260271601976832, 0.9546753343021167, 0.9531268614606743,
    0.7350758766050708, 0.8611979043211514, 0.7948972322338639, 0.8024907962920003, 0.8130873833668756,
    0.9510641655701955, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.9627384084106074, 0.8656950003299526])
reversed_sequence = best_sequence[::-1]
best_sequence = np.concatenate((best_sequence[:-1], reversed_sequence))


def plot_step_function(step_heights_input: list[float]):
    """Plots the step function defined by the given step heights."""
    num_steps = len(step_heights_input)

    # Generate x values for plotting (need to plot steps properly).
    step_edges_plot = np.linspace(-0.25, 0.25, num_steps + 1)
    x_plot = np.array([])
    y_plot = np.array([])

    for i in range(num_steps):
        x_start = step_edges_plot[i]
        x_end = step_edges_plot[i + 1]
        x_step_vals = np.linspace(x_start, x_end, 100)  # Points within each step.
        y_step_vals = np.full_like(x_step_vals, step_heights_input[i])
        x_plot = np.concatenate((x_plot, x_step_vals))
        y_plot = np.concatenate((y_plot, y_step_vals))

    # Plot the step function.
    plt.figure(figsize=(8, 5))
    plt.plot(x_plot, y_plot)
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.title(
        "Step function found by AlphaEvolve for Erdős' minimum overlap problem"
    )
    plt.xlim([-0.3, 0.3])
    plt.ylim([-1, max(step_heights_input) * 1.2])
    plt.grid(True)
    plt.step(
        step_edges_plot[:-1],
        step_heights_input,
        where='post',
        color='green',
        linewidth=2,
    )  # Overlay with plt.step for clarity.
    plt.show()


def verify_sequence(sequence: list[float]):
    """Raises an error if the sequence does not satisfy the constraints."""
    # Check that all values are between 0 and 1.
    if not all(0 <= val <= 1 for val in sequence):
        raise AssertionError("All values in the sequence must be between 0 and 1.")
    # Check that the sum of values in the sequence is exactly n / 2.0.
    if not np.sum(sequence) == len(sequence) / 2.0:
        raise AssertionError(
            "The sum of values in the sequence must be exactly n / 2.0. The sum is "
            f"{np.sum(sequence)} but it should be {len(sequence) / 2.0}.")
    print(
        "The sequence generates a valid step function for Erdős' minimum "
        "overlap problem."
    )


def compute_upper_bound(sequence: list[float]) -> float:
    """Returns the upper bound for a sequence of coefficients."""
    convolution_values = np.correlate(
        np.array(sequence), 1 - np.array(sequence), mode='full'
    )
    return np.max(convolution_values) / len(sequence) * 2


verify_sequence(best_sequence)
plot_step_function(best_sequence)
new_upper_bound = compute_upper_bound(best_sequence)
print(
    f"The sequence provides the following upper bound on the Erdős minimum overlap problem: C5 <= {new_upper_bound}."
)