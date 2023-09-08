import os
import subprocess
import argparse

# Define and parse command line arguments
parser = argparse.ArgumentParser(description='Run distributed inexact policy iteration.')
parser.add_argument('-n', '--populationSize', type=int, required=True, help='Number of population.')
parser.add_argument('-c', '--discountFactor', type=float, required=True, help='Number of actions.') # c for gamma
parser.add_argument('--wf', type=float, required=True, help='weight for financial cost')
parser.add_argument('--wq', type=float, required=True, help='weight for quality of life cost')
parser.add_argument('--wh', type=float, required=True, help='weight for health cost')
#parser.add_argument("--mode", type=str, required=True, choices=["MINCOST", "MAXREWARD"], help="Mode of the problem (MINCOST or MAXREWARD).")
args = parser.parse_args()

# List of CPUs
#cpus = [i for i in range(1, 17)]
cpus = [1, 2, 4, 8, 16]

# Parameters
population = args.populationSize
discountFactor = args.discountFactor
wf = args.wf
wq = args.wq
wh = args.wh
weights = f"{wf},{wq},{wh}"

HM = "0.25,0.125,0.08,0.05,0.03"            # r(a) (hygiene measures)
HM_cf = "0,1,5,6,9"                         # financial cost of hygiene measures
HM_cq = "1,0.7,0.5,0.4,0.05"                # quality of life cost of hygiene measures

#lambdas = [800, 700, 400, 50]
lambdas = [300, 250, 140, 40]
# SD = "20,16,10,1"                         # lambda[a] (social distancing)
SD = f"{lambdas[0]},{lambdas[1]},{lambdas[2]},{lambdas[3]}"
SD_cf = "0,1,10,30"
SD_cq = "1,0.9,0.5,0.1"

mode = "MINCOST"
executable = "./infectious_disease_model"

# Define the directory structure
slurm_id = os.environ["SLURM_JOB_ID"]
dir_output = f"/cluster/home/rosieber/distributed-inexact-policy-iteration/output/IDM/{slurm_id}"

# Options
flags = [
    "-mat_type mpiaij",
    "-pc_type none",
    "-mode", mode,
    "-populationSize", str(population),
    "-discountFactor", str(discountFactor),
    "-weights", weights,
    "-HM", HM,
    "-HM-cf", HM_cf,
    "-HM-cq", HM_cq,
    "-SD", SD,
    "-SD-cf", SD_cf,
    "-SD-cq", SD_cq,
    "-maxIter_PI", str(50),
    "-maxIter_KSP", str(10000),
    "-numPIRuns", str(5),
    "-rtol_KSP", str(1e-5),
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
        "-file_stats", f"{dir_output}/{cpu}/stats.json",
        "-file_policy", f"{dir_output}/{cpu}/policy.out",
        "-file_cost", f"{dir_output}/{cpu}/cost.out"
    ]
    
    # Print the command
    print("[run_benchmark.py] Running command: ")
    print(" ".join(cmd), "\n\n")

    # Run the benchmark
    subprocess.run(cmd)
