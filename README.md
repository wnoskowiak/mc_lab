# Monte Carlo Simulation Script

This script runs a parallel Monte Carlo simulation using MPI inside a Singularity container. You can control the distributions for particle starting positions and travel distances via command-line arguments.

## Requirements

- **OpenMPI**: Required for parallel execution with `mpirun`. Make sure the version installed on your system matches or is compatible with the version used to build the container (e.g., OpenMPI 5.0.5).
- **Singularity**: Required to run the simulation container (`simulation.sif`). Install Singularity on your system to use the provided commands.
- **SLURM**: (Optional) Required if you are running jobs on a cluster using `srun`.

Make sure both OpenMPI and Singularity are properly installed and configured on your system before running the

## container definition (SIngularity.def)

This section was created with ICM systems in mind, where as of the time of writing the installed version of openMPI is 5.0.5. If the system you're planning to run this command on has a different version of MPI installed or the version openMPI on ICM system has changed you'll need to change the definition of the container and rebuild it.

To do so, you need to change values of OPENMPI_VERSION and OPENMPI_MAJOR_VERSION to appropriate values. A simple fix should work fine for mpi versions close to 5.0.5. In case of errors definitions present in this repository can be used as reference:

https://github.com/mfisherman/docker/tree/main/openmpi

## Building the Singularity Container

To build the simulation container, you need to use Singularity and the provided `Singularity.def` file.  
It is recommended to use the `--fakeroot` flag to ensure all necessary permissions during the build process.

Run the following command from your home directory (or the directory containing `Singularity.def`):

```sh
singularity build --fakeroot simulation.sif Singularity.def
```

This will create a `simulation.sif` container image that you can use to run the simulation as described above.

## Usage

Run the script using `srun` and `mpirun` to utilize multiple processes and (optionally) GPUs.  
You can select the distribution for each random variable and pass additional arguments as needed.

### Command-line Arguments

- `--num-particles`: Number of particles for the simulation (default: 10000).
- `--box-length`: Length of the simulation box (default: 33.0).
- `--box-width`: Width of the simulation box (default: 14.0).
- `--box-height`: Height of the simulation box (default: 21.0).
- `--start-x-func`: Distribution for the starting x position (`uniform`, `linear`, `quadratic`)
- `--travel-distance-func`: Distribution for the travel distance (`uniform`, `linear`, `quadratic`)
- `--unit-vector-func`: Distribution for the unit vector direction (`uniform`)
- `--start-x-args`: Arguments for the start x function (e.g., box length)
- `--travel-distance-args`: Arguments for the travel distance function (e.g., max distance)
- `--unit-vector-args`: Arguments for the unit vector function

### Example Command

If your system has mpi natively installed (ie you don't need to schedule a job on a computational node) the srun part of the command can be ommited.

```sh
srun --account=myaccount --mem=150G -N1 -n8 --gres=gpu:1 --time=10:00:00 --pty \
mpirun --mca psec ^munge -n 8 singularity exec ~/mc_lab/simulation.sif python3 ~/mc_lab/src/main.py \
  --num-particles 50000 --box-length 40.0 \
  --start-x-func linear --travel-distance-func quadratic \
  --start-x-args 0 10 \
  --travel-distance-args 1 5 \
  --unit-vector-args -1
```

**Notes:**
- The `-n` flag (number of processes) must be set to the same value in both `srun` and `mpirun`.
- Replace `myaccount` with your actual SLURM account name.
- Adjust memory, time, and other resource requests as needed.
- Paths are set as if everything is in your home directory (`~/mc_lab/...`):

### More Examples

#### 1. All uniform distributions
```sh
srun --account=myaccount --mem=100G -N1 -n4 --gres=gpu:1 --time=02:00:00 --pty \
mpirun --mca psec ^munge -n 4 singularity exec ~/mc_lab/simulation.sif python3 ~/mc_lab/src/main.py \
  --num-particles 20000
```

#### 2. Quadratic start positions, linear travel distance
```sh
srun --account=myaccount --mem=120G -N1 -n6 --gres=gpu:1 --time=04:00:00 --pty \
mpirun --mca psec ^munge -n 6 singularity exec ~/mc_lab/simulation.sif python3 ~/mc_lab/src/main.py \
  --num-particles 30000 --box-length 35.0 \
  --start-x-func quadratic --travel-distance-func linear \
  --start-x-args 0 33 \
  --travel-distance-args 1 10
```

#### 3. Custom arguments for all distributions
```sh
srun --account=myaccount --mem=80G -N1 -n2 --gres=gpu:1 --time=01:00:00 --pty \
mpirun --mca psec ^munge -n 2 singularity exec ~/mc_lab/simulation.sif python3 ~/mc_lab/src/main.py \
  --num-particles 10000 --box-length 50.0 --box-width 20.0 --box-height 25.0 \
  --start-x-func linear --travel-distance-func linear --unit-vector-func uniform \
  --start-x-args 0 50 \
  --travel-distance-args 1 20 \
  --unit-vector-args -1
```

## Notes

- The arguments for `--start-x-args` and `--travel-distance-args` should match the expected parameters for the chosen distribution functions.
- Make sure the Singularity image (`simulation.sif`) and all scripts are in your home directory or adjust the paths accordingly.

---