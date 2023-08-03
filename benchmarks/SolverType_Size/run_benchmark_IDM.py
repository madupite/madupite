import os
import subprocess


# Parameters
discountFactor = 0.7
weights = "5,20,0.05"
HM = "0.25,0.125,0.08,0.05,0.03"
HM_cf = "0,1,5,6,9"
HM_cq = "1,0.7,0.5,0.4,0.05"  

SD_cf = "0,1,10,30"
SD_cq = "1,0.9,0.5,0.1"

mode = "MINCOST"
executable = "./infectious_disease_model"

# Define the directory structure
#slurm_id = "test"
slurm_id = os.environ["SLURM_JOB_ID"]
dir_output = f"/cluster/home/rosieber/distributed-inexact-policy-iteration/output/IDM/SolverType_Size/{slurm_id}"

population_arr = [99, 499, 999, 4999, 9999, 19999, 24999, 39999, 49999]
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
    "-weights", weights,
    "-HM", HM,
    "-HM-cf", HM_cf,
    "-HM-cq", HM_cq,
    "-SD-cf", SD_cf,
    "-SD-cq", SD_cq
]

for solver_name, solver_options in solvers.items():
    for population in population_arr:
        dir = os.path.join(dir_output, solver_name, f"{population+1}") # folder name = numStates for consistency with other models
        os.makedirs(dir, exist_ok=True)
        
        cpu = 16
        cmd = ["mpirun", "-n", str(cpu), "--report-bindings", executable, *flags]

        cmd += [
            "-file_stats", os.path.join(dir, "stats.json"),
            "-file_policy", os.path.join(dir, "policy.out"),
            "-file_cost", os.path.join(dir, "cost.out")
        ]

        cmd += solver_options
        cmd += ["-populationSize", str(population),
                "-SD", f"{int(0.1*population)},{int(0.07*population)},{int(0.05*population)},{int(0.01*population)}"
        ]

        # Print the command
        print("[run_benchmark_GM.py] Running command: ")
        print(" ".join(cmd), "\n\n")

        # Run the benchmark
        subprocess.run(cmd)
