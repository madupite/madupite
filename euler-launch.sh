#!/bin/bash

#SBATCH --output=./data/%j.out
#SBATCH --error=./data/%j.err

#SBATCH --time=0:01:00
#SBATCH --ntasks=6
#SBATCH --mem-per-cpu=20G

mpirun ${HOME}/madupite/build/ci_test;
