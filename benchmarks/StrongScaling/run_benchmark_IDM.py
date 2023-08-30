import os
import subprocess


# Parameters
discountFactor = 0.7
weights = "5,20,0.05"
HM = "0.25,0.125,0.08,0.05,0.03"
HM_cf = "0,1,5,6,9"
HM_cq = "1,0.7,0.5,0.4,0.05"
SD = "500,300,140,80"
SD_cf = "0,1,10,30"
SD_cq = "1,0.9,0.5,0.1"
population = 39999

mode = "MINCOST"
executable = "./infectious_disease_model"

# Define the directory structure
#slurm_id = "test"
slurm_id = os.environ["SLURM_JOB_ID"]
dir_output = f"/cluster/home/rosieber/distributed-inexact-policy-iteration/output/IDM/StrongScaling/{slurm_id}"

#cpus = [1, 2, 4, 8, 16, 24, 32, 40, 48]
cpus = [1, 2, 4, 8, 16, 20, 24, 28, 32, 36, 40, 44, 48]


# general options
flags = [
    "-mat_type", "mpiaij",
    "-pc_type", "none",
    "-maxIter_PI", str(200),
    "-maxIter_KSP", str(10000),
    "-ksp_type", "gmres",
    "-numPIRuns", str(10),
    "-atol_PI", str(1e-10),
    "-rtol_KSP", str(0.1),
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
    "-populationSize", str(population)
]

for cpu in cpus:

    dir = os.path.join(dir_output, f"{cpu}") # folder name = numStates for consistency with other models
    os.makedirs(dir, exist_ok=True)

    cmd = ["mpirun", "-n", str(cpu), "--report-bindings", executable, *flags]

    cmd += [
        "-file_stats", os.path.join(dir, "stats.json"),
        "-file_policy", os.path.join(dir, "policy.out"),
        "-file_cost", os.path.join(dir, "cost.out")
    ]


    # Print the command
    print("[run_benchmark_IDM.py] Running command: ")
    print(" ".join(cmd), "\n\n")

    # Run the benchmark
    subprocess.run(cmd)
