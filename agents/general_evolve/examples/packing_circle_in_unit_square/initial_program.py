# EVOLVE-BLOCK-START
"""Constructor-based circle packing for n=26 circles"""
import sys

import numpy as np


def construct_packing(num_circles=26):
    """
    Construct a specific arrangement of 26 circles in a unit square
    that attempts to maximize the sum of their radii.

    Returns:
        Tuple of (centers, radii, sum_of_radii)
        centers: np.array of shape (26, 2) with (x, y) coordinates
        radii: np.array of shape (26) with radius of each circle
        sum_of_radii: Sum of all radii
    """
    # Initialize arrays for 26 circles
    centers = np.zeros((num_circles, 2))

    # Place circles in a structured pattern
    # This is a simple pattern - evolution will improve this

    # First, place a large circle in the center
    centers[0] = [0.5, 0.5]

    # Place 8 circles around it in a ring
    for i in range(8):
        angle = 2 * np.pi * i / 8
        centers[i + 1] = [0.5 + 0.3 * np.cos(angle), 0.5 + 0.3 * np.sin(angle)]

    # Place 16 more circles in an outer ring
    for i in range(16):
        angle = 2 * np.pi * i / 16
        centers[i + 9] = [0.5 + 0.7 * np.cos(angle), 0.5 + 0.7 * np.sin(angle)]

    # Additional positioning adjustment to make sure all circles
    # are inside the square and don't overlap
    # Clip to ensure everything is inside the unit square
    centers = np.clip(centers, 0.01, 0.99)

    # Compute maximum valid radii for this configuration
    radii = compute_max_radii(centers)

    # Calculate the sum of radii
    sum_radii = np.sum(radii)

    return centers, radii, sum_radii


def compute_max_radii(centers):
    """
    Compute the maximum possible radii for each circle position
    such that they don't overlap and stay within the unit square.

    Args:
        centers: np.array of shape (n, 2) with (x, y) coordinates

    Returns:
        np.array of shape (n) with radius of each circle
    """
    n = centers.shape[0]
    radii = np.ones(n)

    # First, limit by distance to square borders
    for i in range(n):
        x, y = centers[i]
        # Distance to borders
        radii[i] = min(x, y, 1 - x, 1 - y)

    # Then, limit by distance to other circles
    # Each pair of circles with centers at distance d can have
    # sum of radii at most d to avoid overlap
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))

            # If current radii would cause overlap
            if radii[i] + radii[j] > dist:
                # Scale both radii proportionally
                scale = dist / (radii[i] + radii[j])
                radii[i] *= scale
                radii[j] *= scale

    return radii


# EVOLVE-BLOCK-END


# This part remains fixed (not evolved)
def run_packing(num_circles=26):
    """Run the circle packing constructor for n=26"""
    centers, radii, sum_radii = construct_packing(num_circles=num_circles)
    return centers, radii, sum_radii


def _circles_overlap(centers, radii):
    """Protected function to compute max radii."""
    n = centers.shape[0]

    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
            if radii[i] + radii[j] > dist:
                return True

    return False


def check_construction(centers, radii, n) -> dict[str, float]:
    """Evaluates circle packing for maximizing sum of radii in unit square."""

    # General checks for the whole array
    if centers.shape != (n, 2) or not np.isfinite(centers).all():
        print("Error: The 'centers' array has an invalid shape or non-finite values.")
        return {"sum_of_radii": -np.inf}

    # --- Start of the modified geometric check ---

    # 1. Check each circle individually to see if it's contained
    is_contained = ((radii[:, None] <= centers) & (centers <= 1 - radii[:, None])).all(
        axis=1
    )

    # 2. If not all of them are contained...
    if not is_contained.all():
        return {"sum_of_radii": -np.inf}

    if radii.shape != (n,) or not np.isfinite(radii).all() or not (0 <= radii).all():
        print("radii bad shape")
        return {"sum_of_radii": -np.inf}

    if _circles_overlap(centers, radii):
        print("circles overlap")
        return {"sum_of_radii": -np.inf}

    print("The circles are disjoint and lie inside the unit square.")
    return {"sum_of_radii": float(np.sum(radii))}


if __name__ == "__main__":
    centers, radii, sum_radii = run_packing(num_circles=26)
    print(f"Sum of radii: {sum_radii}")

    result_valid = check_construction(centers, radii, 26)
    if result_valid["sum_of_radii"] > -np.inf:
        sys.exit(1)
    else:
        sys.exit(0)
