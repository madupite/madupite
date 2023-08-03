import os
import subprocess


# Parameters
discountFactor = 0.7
weights = "5,20,0.05"
HM = "0.25,0.125,0.08,0.05,0.03"
HM_cf = "0,1,5,6,9"
HM_cq = "1,0.7,0.5,0.4,0.05"  
SD = "200, 160, 100, 20"
SD_cf = "0,1,10,30"
SD_cq = "1,0.9,0.5,0.1"
population = 39000

mode = "MINCOST"
executable = "./infectious_disease_model"

# Define the directory structure
#slurm_id = "test"
slurm_id = os.environ["SLURM_JOB_ID"]
dir_output = f"/cluster/home/rosieber/distributed-inexact-policy-iteration/output/IDM/SolverTolerance/{slurm_id}"

#solver_tolerances = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12]
#solver_tolerances = [1e-2, 1e-4, 1e-8, 1e-10]
solver_tolerances = [0.99, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.01, 0.001, 0.0001, (1-0.7)/(1+0.7)]

kspType = "gmres"


# general options
flags = [
    "-mat_type", "mpiaij",
    "-pc_type", "none",
    "-maxIter_PI", str(100),
    "-maxIter_KSP", str(10000),
    "-numPIRuns", str(20),
    "-atol_PI", str(1e-10),
    "-log_view"
]

flags += [
    "-mode", mode,
    "-discountFactor", str(discountFactor),
    "-weights", weights,
    "-HM", HM,
    "-HM-cf", HM_cf,
    "-HM-cq", HM_cq,
    "-SD-cf", SD_cf,
    "-SD-cq", SD_cq,
    "-SD", SD,
    "-populationSize", str(population),
    "-ksp_type", kspType
]

for tolerance in solver_tolerances:

        dir = os.path.join(dir_output, f"{tolerance:.1e}")
        os.makedirs(dir, exist_ok=True)
        
        cpu = 16
        cmd = ["mpirun", "-n", str(cpu), "--report-bindings", executable, *flags]

        cmd += [
            "-file_stats", os.path.join(dir, "stats.json"),
            "-file_policy", os.path.join(dir, "policy.out"),
            "-file_cost", os.path.join(dir, "cost.out")
        ]

        cmd += ["-rtol_KSP", str(tolerance)]

        # Print the command
        print("[run_benchmark_GM.py] Running command: ")
        print(" ".join(cmd), "\n\n")

        # Run the benchmark
        subprocess.run(cmd)
