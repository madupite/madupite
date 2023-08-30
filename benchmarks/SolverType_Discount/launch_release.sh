#!/bin/bash

#SBATCH -n 16
#SBATCH --time=16:00:00
#SBATCH --job-name="iPI Benchmark"
#SBATCH --mem-per-cpu=8000
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
e=GM

if [ "$e" == "MDP" ]; then
    python ../benchmarks/SolverType_Discount/run_benchmark_MDP.py
    python ../benchmarks/SolverType_Discount/plot.py --path ../output/MDP/SolverType_Discount/$SLURM_JOB_ID


elif [ "$e" == "GM" ]; then
    python ../benchmarks/SolverType_Discount/run_benchmark_GM.py
    python ../benchmarks/SolverType_Discount/plot.py --path ../output/GM/SolverType_Discount/$SLURM_JOB_ID


elif [ "$e" == "IDM" ]; then
    python ../benchmarks/SolverType_Discount/run_benchmark_IDM.py
    python ../benchmarks/SolverType_Discount/plot.py --path ../output/IDM/SolverType_Discount/$SLURM_JOB_ID


elif [ "$e" == "ATC" ]; then
    python ../benchmarks/SolverType_Discount/run_benchmark_ATC.py
    python ../benchmarks/SolverType_Discount/plot.py --path ../output/ATC/SolverType_Discount/$SLURM_JOB_ID


else
    echo "Invalid value of e. It should be either 'MDP', 'GM' or 'IDM'."
    exit 1
fi