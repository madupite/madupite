import os
import subprocess
import argparse

# Define and parse command line arguments
parser = argparse.ArgumentParser(description='Run distributed inexact policy iteration.')
parser.add_argument('-n', '--numStates', type=int, required=True, help='Number of states.')
parser.add_argument('-m', '--numActions', type=int, required=True, help='Number of actions.')
parser.add_argument('-s', '--sparsityFactor', type=float, required=True, help='Sparsity factor.')
#parser.add_argument("--mode", type=str, required=True, choices=["MINCOST", "MAXREWARD"], help="Mode of the problem (MINCOST or MAXREWARD).")
args = parser.parse_args()

# List of CPUs
#cpus = [i for i in range(1, 17)]
ksptype_arr = ["gmres", "tcqmr", "cgs", "tfqmr", "lsqr", "bicg"]

# Parameters
numStates = args.numStates
numActions = args.numActions
sparsityFactor = args.sparsityFactor
mode = "MINCOST"
executable = "./distributed_inexact_policy_iteration"

# Define the directory structure
slurm_id = os.environ["SLURM_JOB_ID"]
dir_scratch = "/cluster/scratch/rosieber/BA_DATA"
dir_data = f"{numStates}_{numActions}_{sparsityFactor:.6f}"
dir_output = f"/cluster/home/rosieber/distributed-inexact-policy-iteration/output/MDP/{slurm_id}"

# Options
flags = [
    "-mat_type mpiaij",
    "-pc_type none",
    "-mode", mode,
    "-states", str(numStates),
    "-actions", str(numActions),
    "-discountFactor", str(0.9),
    "-maxIter_PI", str(50),
    "-maxIter_KSP", str(10000),
    "-numPIRuns", str(20),
    "-rtol_KSP", str(1e-4),
    "-atol_PI", str(1e-10),
    "-log_view"
]

for ksptype in ksptype_arr:
    # Create result directory if it doesn't exist
    os.makedirs(dir_output+f"/{ksptype}/", exist_ok=True)

    # Create a command with options appended
    cpu = 16
    cmd = ["mpirun", "-n", str(cpu), executable, *flags]

    cmd += [
        "-ksp_type", ksptype,
        "-file_P", f"{dir_scratch}/{dir_data}/P.bin",
        "-file_g", f"{dir_scratch}/{dir_data}/g.bin",
        "-file_stats", f"{dir_output}/{ksptype}/stats.json",
        "-file_policy", f"{dir_output}/{ksptype}/policy.out",
        "-file_cost", f"{dir_output}/{ksptype}/cost.out"
    ]
    
    # Print the command
    print("[run_benchmark_MDP.py] Running command: ")
    print(" ".join(cmd), "\n\n")

    # Run the benchmark
    subprocess.run(cmd)