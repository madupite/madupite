import os
import subprocess


# Parameters
riskAversion = 0.5
numK = 10000
numZ = 2
mode = "MAXREWARD"
executable = "./growth_model"

# Define the directory structure
#slurm_id = "test"
slurm_id = os.environ["SLURM_JOB_ID"]
dir_output = f"/cluster/home/rosieber/distributed-inexact-policy-iteration/output/GM/SolverType_Discount/{slurm_id}"

rtol = 0.1
discount_arr = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999]
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
    "-numK", str(numK),
    "-numZ", str(numZ),
    "-riskAversion", str(riskAversion)
]

for solver_name, solver_options in solvers.items():
    for discountFactor in discount_arr:
        dir = os.path.join(dir_output, solver_name, f"{discountFactor:0.6f}")
        os.makedirs(dir, exist_ok=True)
        
        cpu = 16
        cmd = ["mpirun", "-n", str(cpu), "--report-bindings", executable, *flags]

        cmd += [
            "-file_stats", os.path.join(dir, "stats.json"),
            "-file_policy", os.path.join(dir, "policy.out"),
            "-file_cost", os.path.join(dir, "cost.out")
        ]

        cmd += solver_options
        cmd += ["-discountFactor", str(discountFactor)]

        # Print the command
        print("[run_benchmark_GM.py] Running command: ")
        print(" ".join(cmd), "\n\n")

        # Run the benchmark
        subprocess.run(cmd)

