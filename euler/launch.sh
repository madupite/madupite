#!/bin/bash

#SBATCH -n 16
#SBATCH --time=02:00:00
#SBATCH --job-name="iPI Benchmark"
#SBATCH --mem-per-cpu=8192
#SBATCH --mail-type=BEGIN,END

module load gcc/9.3.0
module load cmake/3.25.0
module load openmpi/4.1.4
module load openblas/0.3.20
module load petsc/3.15.5
module load python/3.11.2

cd ../build
make

python ../run_benchmark.py
