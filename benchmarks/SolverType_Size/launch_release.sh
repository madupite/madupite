#!/bin/bash

#SBATCH -n 16
#SBATCH --time=16:00:00
#SBATCH --job-name="iPI Benchmark"
#SBATCH --mem-per-cpu=6000
#SBATCH --mail-type=BEGIN,END

date

module purge
module load gcc/9.3.0
module load cmake/3.25.0
module load openmpi/4.1.4
module load openblas/0.3.20
module load petsc/3.15.5
module load python/3.11.2
module load boost/1.74.0
module list

lscpu

cd ../../release
make

# Variables
e=IDM

if [ "$e" == "MDP" ]; then
    n=18000
    m=40
    s=0.005
    python ../euler/SolverType/run_benchmark_MDP.py -n $n -m $m -s $s
    #python ../plot/strong_scaling.py --path ../output/MDP/$SLURM_JOB_ID/


elif [ "$e" == "GM" ]; then
    python ../euler/SolverType_Size/run_benchmark_GM.py
    #python ../plot/strong_scaling.py --path ../output/GM/$SLURM_JOB_ID/
    python ../plot/SolverType_GM.py --path ../output/SolverType_Size/$SLURM_JOB_ID/


elif [ "$e" == "IDM" ]; then
    python ../euler/SolverType_Size/run_benchmark_IDM.py
    #python ../plot/strong_scaling.py --path ../output/IDM/$SLURM_JOB_ID/


else
    echo "Invalid value of e. It should be either 'MDP', 'GM' or 'IDM'."
    exit 1
fi

