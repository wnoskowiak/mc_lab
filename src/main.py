import argparse
from mpi4py import MPI
from dotenv import load_dotenv
import os
from simulation import monte_carlo_simulation
from simulationCuda import monte_carlo_simulation as monte_carlo_simulation_cuda
import random_functions

# Load environment variables from .env file
load_dotenv()

# Map string names to functions from random_functions
START_X_FUNCTIONS = {
    "uniform": random_functions.random_start_x_uniform,
    "linear": random_functions.random_start_x_linear,
    "quadratic": random_functions.random_start_x_quadratic,
}
TRAVEL_DISTANCE_FUNCTIONS = {
    "uniform": random_functions.random_travel_distance_uniform,
    "linear": random_functions.random_travel_distance_linear,
    "quadratic": random_functions.random_travel_distance_quadratic,
}
UNIT_VECTOR_FUNCTIONS = {
    "uniform": random_functions.random_unit_vectors_uniform,
}

def main(num_particles, box_length, box_width, box_height, start_x_func, travel_distance_func, unit_vector_func, start_x_args, travel_distance_args, unit_vector_args):
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Divide the work among processes
    particles_per_process = num_particles // size
    if rank == size - 1:
        particles_per_process += num_particles % size

    simulation_func = monte_carlo_simulation

    # Run simulation for each process
    exited_particles = simulation_func(
        particles_per_process, box_length, box_width, box_height,
        start_x_func, travel_distance_func, unit_vector_func,
        start_x_args, travel_distance_args, unit_vector_args
    )

    # Gather results from all processes
    total_exited_particles = comm.reduce(exited_particles, op=MPI.SUM, root=0)

    # Print the result in the root process
    if rank == 0:
        print(f"Number of particles that exited the box: {total_exited_particles}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run Monte Carlo simulation.')
    parser.add_argument('--num-particles', type=int, default=10000, help='Number of particles')
    parser.add_argument('--box-length', type=float, default=33.0, help='Length of the box')
    parser.add_argument('--box-width', type=float, default=14.0, help='Width of the box')
    parser.add_argument('--box-height', type=float, default=21.0, help='Height of the box')
    parser.add_argument('--start-x-func', choices=START_X_FUNCTIONS.keys(), default='uniform', help='Distribution for start_x')
    parser.add_argument('--travel-distance-func', choices=TRAVEL_DISTANCE_FUNCTIONS.keys(), default='uniform', help='Distribution for travel_distance')
    parser.add_argument('--unit-vector-func', choices=UNIT_VECTOR_FUNCTIONS.keys(), default='uniform', help='Distribution for unit_vector')
    parser.add_argument('--start-x-args', nargs='*', type=float, default=[], help='Arguments for the start_x function')
    parser.add_argument('--travel-distance-args', nargs='*', type=float, default=[], help='Arguments for the travel_distance function')
    parser.add_argument('--unit-vector-args', nargs='*', type=float, default=[], help='Arguments for the unit_vector function')
    args = parser.parse_args()

    main(
        args.num_particles,
        args.box_length,
        args.box_width,
        args.box_height,
        START_X_FUNCTIONS[args.start_x_func],
        TRAVEL_DISTANCE_FUNCTIONS[args.travel_distance_func],
        UNIT_VECTOR_FUNCTIONS[args.unit_vector_func],
        args.start_x_args,
        args.travel_distance_args,
        args.unit_vector_args
    )