""" Best result """


import numpy as np
import scipy as sp


# Best for evolux
construction_1 = np.array([
    [-2.47590434, -1.81649924],
    [-1.82591428, -2.5764421 ],
    [-0.85258231, -2.80584592],
    [ 0.10902702, -2.53142311],
    [-2.69147575, -0.84001093],
    [-1.53894557, -1.46705928],
    [-0.6094355,  -1.83585601],
    [ 0.7386662,  -1.75453016],
    [-2.33812944,  0.0954893 ],
    [-1.51161838, -0.46743271],
    [-0.54005622, -0.23064788],
    [ 0.02020146, -1.05896644],
    [-1.51863175,  0.66857178],
    [-0.52370935,  0.76921853],
    [ 0.34859503,  0.28025479],
    [ 0.88789215, -0.56186169]])


def run():
    """ Run best solution and print results """

    print(f'Construction 1 has {len(construction_1)} points in {construction_1.shape[1]} dimensions.')
    pairwise_distances = sp.spatial.distance.pdist(construction_1)
    min_distance = np.min(pairwise_distances)
    max_distance = np.max(pairwise_distances)
    ratio_squared = (max_distance / min_distance)**2
    print(f"Ratio of max distance to min distance: sqrt({ratio_squared})")


if __name__ == '__main__':
    run()