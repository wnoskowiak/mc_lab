from numba import cuda
import numpy as np
from scipy.stats import special_ortho_group

@cuda.jit
def monte_carlo_simulation_kernel(num_particles, box_length, box_width, box_height, start_x, travel_distances, directions, results):
    idx = cuda.grid(1)
    if idx < num_particles:
        start_x_val = start_x[idx]
        travel_distance = travel_distances[idx]
        direction = directions[idx]

        # Calculate new position
        end_x = start_x_val + direction[0] * travel_distance
        end_y = box_width / 2 + direction[1] * travel_distance
        end_z = box_height / 2 + direction[2] * travel_distance

        # Check if the particle exits the box
        if end_x < 0 or end_x > box_length or end_y < 0 or end_y > box_width or end_z < 0 or end_z > box_height:
            results[idx] = 1
        else:
            results[idx] = 0

def monte_carlo_simulation(num_particles, box_length, box_width, box_height, start_x_func, travel_distance_func, random_unit_vectors, start_x_args, travel_distance_args, unit_vector_args):
    start_x = start_x_func(num_particles, *start_x_args)
    travel_distances = travel_distance_func(num_particles, *travel_distance_args)
    directions = random_unit_vectors(num_particles, *unit_vector_args)
    results = np.zeros(num_particles, dtype=np.int32)

    # Allocate device memory
    d_start_x = cuda.to_device(start_x)
    d_travel_distances = cuda.to_device(travel_distances)
    d_directions = cuda.to_device(directions)
    d_results = cuda.to_device(results)

    # Configure the blocks
    threads_per_block = 256
    blocks_per_grid = (num_particles + (threads_per_block - 1)) // threads_per_block

    # Launch the kernel
    monte_carlo_simulation_kernel[blocks_per_grid, threads_per_block](num_particles, box_length, box_width, box_height, d_start_x, d_travel_distances, d_directions, d_results)

    # Copy the results back to the host
    results = d_results.copy_to_host()

    return np.sum(results)