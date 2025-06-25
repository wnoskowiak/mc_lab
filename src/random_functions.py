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

# Linear distribution: PDF increases linearly with x in [0, box_length]
def random_start_x_linear(num_particles, *args):
    box_length = args[0]
    u = np.random.uniform(0, 1, num_particles)
    # Inverse CDF: x = box_length * sqrt(u)
    return box_length * np.sqrt(u)

# Quadratic distribution: PDF increases quadratically with x in [0, box_length]
def random_start_x_quadratic(num_particles, *args):
    box_length = args[0]
    u = np.random.uniform(0, 1, num_particles)
    # Inverse CDF: x = box_length * u**(1/3)
    return box_length * np.cbrt(u)

# Linear distribution for travel distance: PDF increases linearly with d in [0, max_distance]
def random_travel_distance_linear(num_particles, *args):
    max_distance = args[0]
    u = np.random.uniform(0, 1, num_particles)
    # Inverse CDF: d = max_distance * sqrt(u)
    return max_distance * np.sqrt(u)

# Quadratic distribution for travel distance: PDF increases quadratically with d in [0, max_distance]
def random_travel_distance_quadratic(num_particles, *args):
    max_distance = args[0]
    u = np.random.uniform(0, 1, num_particles)
    # Inverse CDF: d = max_distance * u**(1/3)
    return max_distance * np.cbrt(u)