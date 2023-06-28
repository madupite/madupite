import os
import subprocess
import argparse

# Define and parse command line arguments
parser = argparse.ArgumentParser(description='Run distributed inexact policy iteration.')
parser.add_argument('-n', '--numStates', type=int, required=True, help='Number of states.')
parser.add_argument('-m', '--numActions', type=int, required=True, help='Number of actions.')
parser.add_argument('-s', '--sparsityFactor', type=float, required=True, help='Sparsity factor.')
args = parser.parse_args()

# List of CPUs
cpus = [1, 2, 4, 8, 16]

# Parameters
numStates = args.numStates
numActions = args.numActions
sparsityFactor = args.sparsityFactor
executable = "./distributed_inexact_policy_iteration"

# Define the directory structure
slurm_id = os.environ["SLURM_JOB_ID"]
dir_scratch = "/cluster/scratch/rosieber/data"
dir_data = f"{numStates}_{numActions}_{sparsityFactor:.6f}"
dir_output = f"/cluster/home/rosieber/distributed-inexact-policy-iteration/output/{slurm_id}"

# Options
flags = [
    "-mat_type mpiaij",
    "-pc_type none",
    "-states", str(numStates),
    "-actions", str(numActions),
    "-discountFactor", str(0.9),
    "-maxIter_PI", str(50),
    "-maxIter_KSP", str(10000),
    "-rtol_KSP", str(1e-4),
    "-atol_PI", str(1e-10),
    "-log_view"
]

# Loop through all CPUs
for cpu in cpus:
    # Create result directory if it doesn't exist
    os.makedirs(dir_output+f"/{cpu}/", exist_ok=True)

    # Create a command with options appended
    cmd = ["mpirun", "-n", str(cpu), executable, *flags]

    cmd += [
        f"-file_P", f"{dir_scratch}/{dir_data}/P.bin",
        f"-file_g", f"{dir_scratch}/{dir_data}/g.bin",
        f"-file_stats", f"{dir_output}/{cpu}/stats.json",
        f"-file_policy", f"{dir_output}/{cpu}/policy.out",
        f"-file_cost", f"{dir_output}/{cpu}/cost.out"
    ]
    
    # Print the command
    print("[run_benchmark.py] Running command: ")
    print(" ".join(cmd), "\n\n")

    # Run the benchmark
    subprocess.run(cmd)