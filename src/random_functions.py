import numpy as np
from scipy.stats import special_ortho_group

def random_unit_vectors_uniform(num_particles, *args):
    vectors = special_ortho_group.rvs(3, size=num_particles)
    return vectors[:, :, 0]

def random_start_x_uniform(num_particles, *args):
    box_length = args[0]
    return np.random.uniform(0, box_length, num_particles)

def random_travel_distance_uniform(num_particles, *args):
    max_distance = args[0]
    return np.random.uniform(0, max_distance, num_particles)  # Arbitrary max distance