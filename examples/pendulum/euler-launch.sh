#!/bin/bash

#SBATCH --output=./data/%j.out
#SBATCH --error=./data/%j.err

#SBATCH --time=0:01:00
#SBATCH --ntasks=6
#SBATCH --mem-per-cpu=20G

mpirun python compute_pendulum.py;
# Make sure to call sbatch from <...>/madupite/examples/pendulum/ to have the correct relative filenames