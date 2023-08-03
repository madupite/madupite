import os
import subprocess


# Parameters
discountFactor = 0.99
mode = "MINCOST"
executable = "./distributed_inexact_policy_iteration"

flight_id = 0
if flight_id == 0:
    states = 260809
    actions = 75
elif flight_id == 1:
    states = 260665
    actions = 75
else:
    raise ValueError(f"Invalid flight id: {flight_id}")


# Define the directory structure
#slurm_id = "test"
slurm_id = os.environ["SLURM_JOB_ID"]
dir_output = f"/cluster/home/rosieber/distributed-inexact-policy-iteration/output/ATC/SolverType/{slurm_id}"

numK_arr = [100, 500, 1000, 5000, 10000, 20000, 25000]
solvers = {
    "gmres" : [
        "-ksp_type",  "gmres",
        "-maxIter_KSP", str(10000),
        "-rtol_KSP", str(1e-4)
    ],
    "opi50" : [
        "-ksp_type", "richardson",
        "-maxIter_KSP", str(50),
        "-rtol_KSP", str(1e-20),
        "-ksp_richardson_scale", "1.0"
    ],
    "opi100" : [
        "-ksp_type", "richardson",
        "-maxIter_KSP", str(100),
        "-rtol_KSP", str(1e-20),
        "-ksp_richardson_scale", "1.0"
    ],
    "opi500" : [
        "-ksp_type", "richardson",
        "-maxIter_KSP", str(500),
        "-rtol_KSP", str(1e-20),
        "-ksp_richardson_scale", "1.0"
    ],
    "cgs" : [
        "-ksp_type", "cgs",
        "-maxIter_KSP", str(10000),
        "-rtol_KSP", str(1e-4)
    ],
    "tfqmr" : [
        "-ksp_type", "tfqmr",
        "-maxIter_KSP", str(10000),
        "-rtol_KSP", str(1e-4)
    ],
    "bicg" : [
        "-ksp_type", "bicg",
        "-maxIter_KSP", str(10000),
        "-rtol_KSP", str(1e-4)
    ]
}


# general options
flags = [
    "-mat_type mpiaij",
    "-pc_type none",
    "-maxIter_PI", str(200),
    "-numPIRuns", str(10),
    "-atol_PI", str(1e-10),
    "-log_view"
]

flags += [
    "-mode", mode,
    "-discountFactor", str(discountFactor),
    "-states", str(states),
    "-actions", str(actions),
    "-file_P", f"/cluster/scratch/rosieber/BA_DATA/ATC/data/{flight_id}_P.bin",
    "-file_g", f"/cluster/scratch/rosieber/BA_DATA/ATC/data/{flight_id}_g.bin"
]

for solver_name, solver_options in solvers.items():
    dir = os.path.join(dir_output, solver_name) # folder name = numStates for consistency with other models
    os.makedirs(dir, exist_ok=True)
    
    cpu = 16
    cmd = ["mpirun", "-n", str(cpu), "--report-bindings", executable, *flags]

    cmd += [
        "-file_stats", os.path.join(dir, "stats.json"),
        "-file_policy", os.path.join(dir, "policy.out"),
        "-file_cost", os.path.join(dir, "cost.out")
    ]

    cmd += solver_options

    # Print the command
    print("[run_benchmark_GM.py] Running command: ")
    print(" ".join(cmd), "\n\n")

    # Run the benchmark
    subprocess.run(cmd)
