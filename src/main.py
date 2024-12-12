import argparse
from mpi4py import MPI
from dotenv import load_dotenv
import os
from simulation import monte_carlo_simulation
from simulationCuda import monte_carlo_simulation as monte_carlo_simulation_cuda
from random_functions import random_unit_vectors_uniform, random_start_x_uniform, random_travel_distance_uniform

# Load environment variables from .env file
load_dotenv()

# Read parameters from environment variables
num_particles = int(os.getenv('NUM_PARTICLES', 10000))
box_length = float(os.getenv('BOX_LENGTH', 33))
box_width = float(os.getenv('BOX_WIDTH', 14))
box_height = float(os.getenv('BOX_HEIGHT', 21))

def main(use_cuda, start_x_args, travel_distance_args, unit_vector_args):
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Divide the work among processes
    particles_per_process = num_particles // size
    if rank == size - 1:
        particles_per_process += num_particles % size

    # Choose the appropriate simulation function
    if use_cuda:
        simulation_func = monte_carlo_simulation_cuda
    else:
        simulation_func = monte_carlo_simulation

    # Run simulation for each process
    exited_particles = simulation_func(
        particles_per_process, box_length, box_width, box_height,
        random_start_x_uniform, random_travel_distance_uniform, random_unit_vectors_uniform,
        start_x_args, travel_distance_args, unit_vector_args
    )

    # Gather results from all processes
    total_exited_particles = comm.reduce(exited_particles, op=MPI.SUM, root=0)

    # Print the result in the root process
    if rank == 0:
        print(f"Number of particles that exited the box: {total_exited_particles}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run Monte Carlo simulation.')
    parser.add_argument('--use-cuda', action='store_true', help='Use CUDA for simulation')
    parser.add_argument('--start-x-args', nargs='*', type=float, default=[], help='Arguments for the start_x function')
    parser.add_argument('--travel-distance-args', nargs='*', type=float, default=[], help='Arguments for the travel_distance function')
    parser.add_argument('--unit-vector-args', nargs='*', type=float, default=[], help='Arguments for the unit_vector function')
    args = parser.parse_args()

    main(args.use_cuda, args.start_x_args, args.travel_distance_args, args.unit_vector_args)