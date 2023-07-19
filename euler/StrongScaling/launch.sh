#!/bin/bash

#SBATCH -n 16
#SBATCH --time=05:00:00
#SBATCH --job-name="iPI Benchmark"
#SBATCH --mem-per-cpu=4096
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

cd ../build
make

# Variables
e=GM

if [ "$e" == "MDP" ]; then
    n=15000
    m=100
    s=0.003
    python ../euler/StrongScaling/run_benchmark_MDP.py -n $n -m $m -s $s
    python ../plot/strong_scaling.py --path ../output/MDP/$SLURM_JOB_ID/


elif [ "$e" == "GM" ]; then
    k=8000
    r=0.5
    python ../euler/StrongScaling/run_benchmark_GM.py --numK $k --riskAversion $r
    python ../plot/strong_scaling.py --path ../output/GM/$SLURM_JOB_ID/


elif [ "$e" == "IDM" ]; then
    n=5000
    c=0.7
    wf=2
    wq=10
    wh=0.2
    python ../euler/StrongScaling/run_benchmark_IDM.py -n $n --discountFactor $c --wf $wf --wq $wq --wh $wh
    python ../plot/strong_scaling.py --path ../output/IDM/$SLURM_JOB_ID/


else
    echo "Invalid value of e. It should be either 'MDP', 'GM' or 'IDM'."
    exit 1
fi

