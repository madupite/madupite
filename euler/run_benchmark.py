import os
import subprocess

# List of CPUs
cpus = [1, 2, 4, 8, 16]

# Parameters
numStates = 9000
numActions = 100
sparsityFactor = 0.1
executable = "./distributed_inexact_policy_iteration"

# Define the directory structure
slurm_id = os.environ["SLURM_JOB_ID"]
dir_scratch = f"/cluster/scratch/rosieber/data"
dir_data = f"{numStates}_{numActions}_{sparsityFactor:.6f}"
dir_output = f"/cluster/home/rosieber/distributed-inexact-policy-iteration/output/{slurm_id}"

# Options
flags = {
    "-mat_type": "mpiaij",
    "-pc_type": "none",
    "-states": str(numStates),
    "-actions": str(numActions),
    "-discountFactor": "0.9",
    "-maxIter_PI": "50",
    "-maxIter_KSP": "10000",
    "-rtol_KSP": "1e-4",
    "-atol_PI": "1e-10",
}

# Loop through all CPUs
for cpu in cpus:
    # Create result directory if it doesn't exist
    os.makedirs(dir_output+f"/{cpu}/", exist_ok=True)

    # Create a command with options appended
    cmd = ["mpirun", "-n", str(cpu), executable]

    for option, value in flags.items():
        cmd.extend([option, value])
    
    cmd.extend([
        "-file_P", f"{dir_scratch}/{dir_data}/P.bin",
        "-file_g", f"{dir_scratch}/{dir_data}/g.bin",
        "-file_stats", f"{dir_output}/{cpu}/stats.json",
        "-file_policy", f"{dir_output}/{cpu}/policy.out",
        "-file_cost", f"{dir_output}/{cpu}/cost.out",
    ])
    

    # Print the command
    print("[run_benchmark.py] Running command: ")
    print(" ".join(cmd), "\n\n")

    # Run the benchmark
    subprocess.run(cmd)