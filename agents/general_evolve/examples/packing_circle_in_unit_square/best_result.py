"""
LoongFlow - Circle Packing best result
"""

import numpy as np
import itertools
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def verify_circles(circles: np.ndarray):
    """Checks that the circles are disjoint and lie inside a unit square.

    Args:
      circles: A numpy array of shape (num_circles, 3), where each row is
        of the form (x, y, radius), specifying a circle.

    Raises:
      AssertionError if the circles are not disjoint or do not lie inside the
      unit square.
    """
    # Check pairwise disjointness.
    for circle1, circle2 in itertools.combinations(circles, 2):
        center_distance = np.sqrt(
            (circle1[0] - circle2[0]) ** 2 + (circle1[1] - circle2[1]) ** 2
        )
        radii_sum = circle1[2] + circle2[2]
        assert (
            center_distance >= radii_sum
        ), f"Circles are NOT disjoint: {circle1} and {circle2}."

    # Check all circles lie inside the unit square [0,1]x[0,1].
    for circle in circles:
        assert (
            0 <= min(circle[0], circle[1]) - circle[2]
            and max(circle[0], circle[1]) + circle[2] <= 1
        ), f"Circle {circle} is NOT fully inside the unit square."


def plot_circles(circles: np.ndarray):
    """Plots the circles."""
    _, ax = plt.subplots(1, figsize=(7, 7))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")  # Make axes scaled equally.

    # Draw unit square boundary.
    rect = patches.Rectangle(
        (0, 0), 1, 1, linewidth=1, edgecolor="black", facecolor="none"
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
        f"A collection of {len(circles)} disjoint circles packed inside a unit square to maximize the sum of radii"
    )
    plt.show()


construction_1 = np.array(
    [
        [0.0849263083039182, 0.9150736917086635, 0.0849254214866789],
        [0.5976350262144691, 0.2716296931970270, 0.0998976849114993],
        [0.6820783857757839, 0.9038494574875026, 0.0961504613342175],
        [0.5299633910221666, 0.4986677487238165, 0.1370098772227197],
        [0.3869265056297526, 0.2947454329076524, 0.1120739974543406],
        [0.7629585602245132, 0.7593516667966225, 0.0694394160300840],
        [0.0788604411074772, 0.4972835295839336, 0.0788600859932671],
        [0.9074077718310999, 0.6859431949917751, 0.0925916131076377],
        [0.9076084527893614, 0.3131157586756130, 0.0923910466338807],
        [0.7424170368199461, 0.4033586590488636, 0.0958418105166246],
        [0.9060726187805146, 0.4994283583972048, 0.0939268831642502],
        [0.2753422276006534, 0.4955311833386117, 0.1176285145990125],
        [0.4846008453615003, 0.1030605748871513, 0.1030600768913956],
        [0.2747834528766626, 0.1067893155740154, 0.1067893155740154],
        [0.4825946797939218, 0.8965328427880785, 0.1034663780562484],
        [0.5960422232148862, 0.7269056532012732, 0.1006000507755823],
        [0.7420492496374718, 0.5952196251570064, 0.0960181176332300],
        [0.7636736200287229, 0.2397104547207950, 0.0691801500168185],
        [0.8888441894438750, 0.8888441894497398, 0.1111558105502602],
        [0.6832586089510956, 0.0957323045987883, 0.0957318067333429],
        [0.3816653253049620, 0.7026093496635539, 0.1151482591261624],
        [0.0846397134603101, 0.0846397134380425, 0.0846391662568031],
        [0.1332587149888469, 0.7023091624587523, 0.1332575325706433],
        [0.8892210117182656, 0.1107789882394257, 0.1107784860382811],
        [0.1302193912915700, 0.2946091498077932, 0.1302193912915700],
        [0.2739524527358436, 0.8948181047812457, 0.1051818952187543],
    ]
)

print(f"Construction 1 has {len(construction_1)} circles.")
verify_circles(construction_1)
print(f"The circles are disjoint and lie inside the unit square.")
sum_radii = np.sum(construction_1[:, 2])
print(f"Construction 1 sum of radii: {sum_radii}")

plot_circles(construction_1)
