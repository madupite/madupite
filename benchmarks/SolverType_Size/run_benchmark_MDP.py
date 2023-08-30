import os
import subprocess


# Parameters
discountFactor = 0.99
mode = "MINCOST"
executable = "./distributed_inexact_policy_iteration"

# Define the directory structure
#slurm_id = "test"
slurm_id = os.environ["SLURM_JOB_ID"]
data_dir = f"/cluster/scratch/rosieber/BA_DATA"
dir_output = f"/cluster/home/rosieber/distributed-inexact-policy-iteration/output/MDP/SolverType_Size/{slurm_id}"

states_arr = [100, 500, 1000, 5000, 10000, 20000, 30000]
actions_arr = [5, 10, 25, 50, 100, 200]
sparsityFactor = 0.005

rtol = 0.01
solvers = {
    "gmres" : [
        "-ksp_type",  "gmres",
        "-maxIter_KSP", str(10000),
        "-rtol_KSP", str(rtol)
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
    "tfqmr" : [
        "-ksp_type", "tfqmr",
        "-maxIter_KSP", str(10000),
        "-rtol_KSP", str(rtol)
    ],
    "bcgs" : [
        "-ksp_type", "bcgs",
        "-maxIter_KSP", str(10000),
        "-rtol_KSP", str(rtol)
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
]



for solver_name, solver_options in solvers.items():
    for numStates in states_arr:
        for numActions in actions_arr:
            dir = os.path.join(dir_output, solver_name, str(numStates), str(numActions)) # solver/state/action
            os.makedirs(dir, exist_ok=True)

            cpu = 16
            cmd = ["mpirun", "-n", str(cpu), "--report-bindings", executable, *flags]

            cmd += [
                "-file_P", f"{data_dir}/P_{numStates}_{numActions}_{sparsityFactor:0.6f}.bin",
                "-file_g", f"{data_dir}/g_{numStates}_{numActions}_{sparsityFactor:0.6f}.bin",
                "-file_stats", os.path.join(dir, "stats.json"),
                "-file_policy", os.path.join(dir, "policy.out"),
                "-file_cost", os.path.join(dir, "cost.out")
            ]

            cmd += solver_options
            cmd += ["-states", str(numStates), "-actions", str(numActions)]

            # Print the command
            print("[run_benchmark_MDP.py] Running command: ")
            print(" ".join(cmd), "\n\n")

            # Run the benchmark
            subprocess.run(cmd)
