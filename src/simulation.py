import numpy as np
from scipy.stats import special_ortho_group

def monte_carlo_simulation(num_particles, box_length, box_width, box_height, start_x_func, travel_distance_func, random_unit_vectors, start_x_args, travel_distance_args, unit_vector_args):
    # Generate all random numbers in one go
    start_x = start_x_func(num_particles, *start_x_args)
    travel_distances = travel_distance_func(num_particles, *travel_distance_args)
    directions = random_unit_vectors(num_particles, *unit_vector_args)

    # Particle starts at a random point on the line of length 33 cm going through the middle of the box
    start_positions = np.column_stack((start_x, np.full(num_particles, box_width / 2), np.full(num_particles, box_height / 2)))

    # Calculate new positions
    end_positions = start_positions + directions * travel_distances[:, np.newaxis]

    # Check if the particles exit the box
    exited_particles = np.any(end_positions < 0, axis=1) | np.any(end_positions > [box_length, box_width, box_height], axis=1)

    # Count the number of particles that exited the box
    particles_exited = np.sum(exited_particles)

    return particles_exited