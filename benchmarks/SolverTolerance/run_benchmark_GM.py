import os
import subprocess


# Parameters
riskAversion = 0.5
discountFactor = 0.98
numZ = 2
numK = 4000
mode = "MAXREWARD"
executable = "./growth_model"

# Define the directory structure
#slurm_id = "test"
slurm_id = os.environ["SLURM_JOB_ID"]
dir_output = f"/cluster/home/rosieber/distributed-inexact-policy-iteration/output/GM/SolverTolerance/{slurm_id}"

#solver_tolerances = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12]
#solver_tolerances = [1e-2, 1e-4, 1e-8, 1e-10]
solver_tolerances = [0.99, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.01, 0.001, 0.0001, (1-0.98)/(1+0.98)]

kspType = "gmres"


# general options
flags = [
    "-mat_type", "mpiaij",
    "-pc_type", "none",
    "-maxIter_PI", str(100),
    "-maxIter_KSP", str(10000),
    "-numPIRuns", str(20),
    "-atol_PI", str(1e-10),
    "-log_view"
]

flags += [
    "-mode", mode,
    "-discountFactor", str(discountFactor),
    "-numZ", str(numZ),
    "-numK", str(numK),
    "-ksp_type", kspType,
    "-riskAversion", str(riskAversion)
]

for tolerance in solver_tolerances:

        dir = os.path.join(dir_output, f"{tolerance:.1e}")
        os.makedirs(dir, exist_ok=True)
        
        cpu = 16
        cmd = ["mpirun", "-n", str(cpu), "--report-bindings", executable, *flags]

        cmd += [
            "-file_stats", os.path.join(dir, "stats.json"),
            "-file_policy", os.path.join(dir, "policy.out"),
            "-file_cost", os.path.join(dir, "cost.out")
        ]

        cmd += ["-rtol_KSP", str(tolerance)]

        # Print the command
        print("[run_benchmark_GM.py] Running command: ")
        print(" ".join(cmd), "\n\n")

        # Run the benchmark
        subprocess.run(cmd)
