import os
import subprocess


# Parameters
discountFactor = 0.99
mode = "MINCOST"
executable = "./distributed_inexact_policy_iteration"

# Define the directory structure
#slurm_id = "test"
slurm_id = os.environ["SLURM_JOB_ID"]
data_dir = f"/cluster/scratch/rosieber/BA_DATA/"
dir_output = f"/cluster/home/rosieber/distributed-inexact-policy-iteration/output/MDP/StrongScaling/{slurm_id}"

#states_arr = [100, 500, 1000, 5000, 10000, 15000, 30000, 50000]
#actions_arr = [5, 10, 20, 50, 100, 200]
numStates = 20000
numActions = 100
sparsityFactor = 0.005

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
    "-rtol_KSP", str(0.01),
    "-log_view"
]

flags += [
    "-mode", mode,
    "-discountFactor", str(discountFactor),
    "-states", str(numStates),
    "-actions", str(numActions),
]

for cpu in cpus:

    dir = os.path.join(dir_output, f"{cpu}") # folder name = numStates for consistency with other models
    os.makedirs(dir, exist_ok=True)

    cmd = ["mpirun", "-n", str(cpu), "--report-bindings", executable, *flags]

    cmd += [
        "-file_P", os.path.join(data_dir, f"P_{numStates}_{numActions}_{sparsityFactor:0.6f}.bin"),
        "-file_g", os.path.join(data_dir, f"g_{numStates}_{numActions}_{sparsityFactor:0.6f}.bin"),
        "-file_stats", os.path.join(dir, "stats.json"),
        "-file_policy", os.path.join(dir, "policy.out"),
        "-file_cost", os.path.join(dir, "cost.out")
    ]


    # Print the command
    print("[run_benchmark_MDP.py] Running command: ")
    print(" ".join(cmd), "\n\n")

    # Run the benchmark
    subprocess.run(cmd)
