"""Visualization of 21 Circle Packing"""

# @title Visualization of 21 Circle Packing
import itertools

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

circles = np.array(
    [
        [0.5802742936796795, 0.3984465473580798, 0.13882521515344715],
        [0.18156155272356161, 0.43652064487907954, 0.11292170430401849],
        [0.20571117563521127, 0.18769683629756662, 0.13707127960492216],
        [0.7999784154376184, 0.3488287602048943, 0.08641205067835192],
        [0.7827643894440595, 0.5465250863733603, 0.11203229938326592],
        [0.46121144602322633, 0.1696883947733613, 0.11906284958108318],
        [0.5802743126915707, 0.8999564216611383, 0.12740019460035393],
        [0.19260149781970107, 0.9033950007640802, 0.12396159400193699],
        [0.18372300046198145, 0.6645152241131472, 0.11508312024224436],
        [0.967947166948219, 0.9033949590982809, 0.12396160364412899],
        [0.7754366350452009, 0.9526151108548776, 0.07474151216459557],
        [0.5802743032771106, 0.6549139948297392, 0.11764223223100972],
        [0.9768256202938239, 0.6645151601528155, 0.11508313124448671],
        [0.7773824857710533, 0.7681879617308184, 0.10969590179745901],
        [0.38316613883650913, 0.7681879771929849, 0.10969589384897398],
        [0.37778419905713695, 0.546525102225111, 0.11203230781873637],
        [0.3605701631556298, 0.3488287597735352, 0.08641205932413132],
        [0.699337146134056, 0.16968839617678885, 0.11906285052901798],
        [0.9548374333711164, 0.18769685433053626, 0.1370712966337934],
        [0.38511201030764153, 0.952615094644837, 0.07474148866158556],
        [0.9789869570840519, 0.43652062911558004, 0.11292164405328059],
    ]
)


def minimum_circumscribing_rectangle(circles: np.ndarray) -> tuple[float, float]:
    """Returns the width and height of the minimum circumscribing rectangle.

    Args:
      circles: A numpy array of shape (num_circles, 3), where each row is of the
        form (x, y, radius), specifying a circle.

    Returns:
      A tuple (width, height) of the minimum circumscribing rectangle.
    """
    min_x = np.min(circles[:, 0] - circles[:, 2])
    max_x = np.max(circles[:, 0] + circles[:, 2])
    min_y = np.min(circles[:, 1] - circles[:, 2])
    max_y = np.max(circles[:, 1] + circles[:, 2])
    return max_x - min_x, max_y - min_y


def verify_circles_disjoint(circles: np.ndarray):
    """Checks that circles are disjoint.

    Args:
      circles: A numpy array of shape (num_circles, 3), where each row is of the
        form (x, y, radius), specifying a circle.

    Raises:
      AssertionError: if the circles are not disjoint.
    """
    for circle1, circle2 in itertools.combinations(circles, 2):
        center_distance = np.sqrt(
            (circle1[0] - circle2[0]) ** 2 + (circle1[1] - circle2[1]) ** 2
        )
        radii_sum = circle1[2] + circle2[2]
        assert (
            center_distance >= radii_sum
        ), f"Circles are NOT disjoint: {circle1} and {circle2}."


def plot_circles_rectangle(circles: np.ndarray):
    """Plots the given circles in a rectangle."""
    width, height = minimum_circumscribing_rectangle(circles)
    _, ax = plt.subplots(1, figsize=(7, 7))
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_aspect("equal")  # Make axes scaled equally.

    # Draw rectangle boundary.
    rect = patches.Rectangle(
        (0, 0), width, height, linewidth=1, edgecolor="black", facecolor="none"
    )
    ax.add_patch(rect)

    # Draw the circles.
    for circle in circles:
        circ = patches.Circle(
            (circle[0], circle[1]),
            circle[2],
            edgecolor="blue",
            facecolor="skyblue",
            alpha=0.5,
        )
        ax.add_patch(circ)

    plt.title(
        f"{len(circles)} disjoint circles packed inside a rectangle of perimeter {round(2 * width + 2 * height, 6)}"
    )
    plt.show()


if __name__ == "__main__":
    num_circles = len(circles)
    verify_circles_disjoint(circles)
    print(f"Construction has {num_circles} disjoint circles.")
    width, height = minimum_circumscribing_rectangle(circles)
    print(f"Perimeter of minimum circumscribing rectangle: {2 * (width + height):.6f}")
    assert (
        width + height
    ) <= 2, "The circles cannot be contained within a rectangle with perimeter 4."
    sum_radii = np.sum(circles[:, 2])
    print("Sum of radii:", sum_radii)

    plot_circles_rectangle(circles)
