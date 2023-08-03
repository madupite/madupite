import os
import subprocess
import argparse

# Define and parse command line arguments
parser = argparse.ArgumentParser(description='Run distributed inexact policy iteration.')
parser.add_argument('-k', '--numK', type=int, required=True, help='number of capital stocks.')
parser.add_argument('-r', '--riskAversion', type=float, required=True, help='risk aversion factor.')
args = parser.parse_args()

# List of CPUs
#cpus = [i for i in range(1, 17)]
cpus = [1, 2, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48]

# Parameters
numK = args.numK
riskAversion = args.riskAversion
discountFactor = 0.98
numZ = 2
mode = "MAXREWARD"
executable = "./growth_model"

# Define the directory structure
slurm_id = os.environ["SLURM_JOB_ID"]
dir_output = f"/cluster/home/rosieber/distributed-inexact-policy-iteration/output/GM/{slurm_id}"

# Options
flags = [
    "-mat_type mpiaij",
    "-pc_type none",
    "-mode", mode,
    "-numK", str(numK),
    "-numZ", str(numZ),
    "-riskAversion", str(riskAversion),
    "-discountFactor", str(discountFactor),
    "-maxIter_PI", str(200),
    "-maxIter_KSP", str(10000),
    "-numPIRuns", str(10),
    "-rtol_KSP", str(1e-4),
    "-atol_PI", str(1e-10),
    "-log_view"
]

# Loop through all CPUs
for cpu in cpus:
    # Create result directory if it doesn't exist
    os.makedirs(dir_output+f"/{cpu}/", exist_ok=True)

    # Create a command with options appended
    cmd = ["mpirun", "-n", str(cpu), "--report-bindings", executable, *flags]

    cmd += [
        "-file_stats", f"{dir_output}/{cpu}/stats.json",
        "-file_policy", f"{dir_output}/{cpu}/policy.out",
        "-file_cost", f"{dir_output}/{cpu}/cost.out"
    ]
    
    # Print the command
    print("[run_benchmark_GM.py] Running command: ")
    print(" ".join(cmd), "\n\n")

    # Run the benchmark
    subprocess.run(cmd)
