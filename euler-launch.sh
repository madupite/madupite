#!/bin/bash

#SBATCH --output=./data/%j.out
#SBATCH --error=./data/%j.err

#SBATCH --time=0:01:00
#SBATCH --ntasks=6
#SBATCH --mem-per-cpu=20G

mpirun python ${HOME}/madupite/examples/install/main.py -file_stats stats.json;
