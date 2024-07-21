#!/bin/bash

#SBATCH --ntasks=48
#SBATCH --time=2:00:00
#SBATCH --job-name="strong scaling benchmark"
#SBATCH --mem-per-cpu=2560
#SBATCH --tmp=48000

source ./job.sh

