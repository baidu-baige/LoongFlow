"""Second autocorrelation inequality Problem"""

# @title Visualization
import matplotlib.pyplot as plt
import numpy as np


def plot_step_function(step_heights_input: list[float], title=""):
    """Plots a step function with equally-spaced intervals on [-1/4,1/4]."""
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
    plt.title(title)
    plt.xlim([-0.3, 0.3])  # Adjust x-axis limits if needed.
    plt.ylim([-1, max(step_heights_input) * 1.2])  # Adjust y-axis limits.
    plt.grid(True)
    plt.step(
        step_edges_plot[:-1],
        step_heights_input,
        where="post",
        color="green",
        linewidth=2,
    )
    plt.show()


if __name__ == "__main__":
    height_sequence = [
        0.0000004680105658,
        0.0000612283150516,
        7.5184836962413968,
        15.2353814124068307,
        0.0000023818234039,
        0.0000001917606383,
        0.0000059597698920,
        0.0000161397966633,
        0.0000002966857691,
        0.0000028901162036,
        0.0000038365180752,
        0.0000002151403133,
        0.0000132093028546,
        0.0004780108084031,
        0.0200867077042187,
        4.9771780726701120,
        5.3506742944751426,
        4.5918988636830820,
        5.2380206835472816,
        1.9352845938841128,
        0.0667869581703989,
        0.0009607035357567,
        0.0080780684767165,
        0.7804226267247305,
        1.6339610288143867,
        0.0655294679535946,
        0.0365557678626271,
        4.1891999302922827,
        4.0191044319190512,
        3.7273738386683544,
        2.9254027802483762,
        2.7677579201253213,
        3.4165052053004628,
        3.8925833971164394,
        4.4936314724977331,
        4.9224483485196764,
        4.6386227164932796,
        4.3766496749146988,
        4.8781632606426140,
        3.9244977292205365,
        3.2112243878188029,
        2.7815972918479108,
        1.8484911703326707,
        1.7094920736894843,
        1.8513028355243950,
        1.8167097572103279,
        1.6963632850980599,
        1.3157090492126458,
        0.3570802982717114,
        0.3725108855096496,
    ]

    convolution_2 = np.convolve(height_sequence, height_sequence)

    # @title Verification
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
        interval_l2_squared = (h / 3) * (y1**2 + y1 * y2 + y2**2)
        l2_norm_squared += interval_l2_squared

    # Calculate the 1-norm: ||f*f||_1
    norm_1 = np.sum(np.abs(convolution_2)) / (len(convolution_2) + 1)

    # Calculate the infinity-norm: ||f*f||_inf
    norm_inf = np.max(np.abs(convolution_2))
    C_lower_bound = l2_norm_squared / (norm_1 * norm_inf)

    print(f"This step function shows that C2 >= {C_lower_bound}")

    plot_step_function(height_sequence, title="LoongFlow's discovered step function")
    plot_step_function(
        convolution_2, title="Autoconvolution of the discovered step function"
    )
