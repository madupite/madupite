import os
import subprocess


# Parameters
riskAversion = 0.5
discountFactor = 0.98
numK = 10000
numZ = 2
mode = "MAXREWARD"
executable = "./growth_model"

# Define the directory structure
#slurm_id = "test"
slurm_id = os.environ["SLURM_JOB_ID"]
dir_output = f"/cluster/home/rosieber/distributed-inexact-policy-iteration/output/GM/StrongScaling/{slurm_id}"

#cpus = [1, 2, 4, 8, 16, 24, 32, 40, 48]
cpus = [1, 2, 4, 8, 16, 20, 24, 28, 32, 36, 40, 44, 48]


# general options
flags = [
    "-mat_type", "mpiaij",
    "-pc_type", "none",
    "-maxIter_PI", str(200),
    "-numPIRuns", str(10),
    "-atol_PI", str(1e-10),
    "-log_view"
]

flags += [
    "-mode", mode,
    "-discountFactor", str(discountFactor),
    "-numZ", str(numZ),
    "-numK", str(numK),
    "-riskAversion", str(riskAversion),
    "-ksp_type", "gmres",
    "-rtol_KSP", str(0.1),
    "-maxIter_KSP", str(1000)
]

for cpu in cpus:

    dir = os.path.join(dir_output, f"{cpu}") # folder name = numStates for consistency with other models
    os.makedirs(dir, exist_ok=True)

    cmd = ["mpirun", "-n", str(cpu), "--map-by", f"ppr:{int(cpu/2 + 0.5)}:socket", "--bind-to", "core", "--rank-by", "core", "--report-bindings", executable, *flags]

    cmd += [
        "-file_stats", os.path.join(dir, "stats.json"),
        "-file_policy", os.path.join(dir, "policy.out"),
        "-file_cost", os.path.join(dir, "cost.out")
    ]


    # Print the command
    print("[run_benchmark_GM.py] Running command: ")
    print(" ".join(cmd), "\n\n")

    # Run the benchmark
    subprocess.run(cmd)
