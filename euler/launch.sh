#!/bin/bash

#SBATCH -n 16
#SBATCH --time=01:00:00
#SBATCH --job-name="iPI Benchmark"
#SBATCH --mem-per-cpu=8192
#SBATCH --mail-type=BEGIN,END

module purge
module load gcc/9.3.0
module load cmake/3.25.0
module load openmpi/4.1.4
module load openblas/0.3.20
module load petsc/3.15.5
module load python/3.11.2
module list

lscpu
date

cd ../build
make

# Variables
n=9000
m=100
s=0.1

python ../euler/run_benchmark.py -n $n -m $m -s $s
