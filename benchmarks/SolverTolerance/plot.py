import os
import json
import numpy as np
import matplotlib.pyplot as plt
import argparse

# file path as command line argument
parser = argparse.ArgumentParser(description='Plot runtimes and speedup of iPI')
parser.add_argument('--path', required=True, help='path of the output data (parent directory of cpu directories)')
args = parser.parse_args()
path = args.path


# plot ksp iterations (y) vs pi step (x) for different tolerances
tolerance_dirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
ksp_iterations = {}
runtimes = {}
mean_runtimes = {}

for tolerance_dir in tolerance_dirs:
    ksp_iterations[tolerance_dir] = []
    runtimes[tolerance_dir] = []
    mean_runtimes[tolerance_dir] = []
    json_path = os.path.join(path, tolerance_dir, 'stats.json')
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Get iteration data
    for iteration in data["solver_runs"][0]:
        ksp_iterations[tolerance_dir].append(iteration["ksp_iterations"])

    # compute mean runtime
    for run in data['solver_runs']:
        total_runtime = sum([iteration['computation_time'] for iteration in run])
        runtimes[tolerance_dir].append(total_runtime / 1000)
    mean_runtimes[tolerance_dir] = np.mean(runtimes[tolerance_dir])


# Plotting
plt.figure(figsize=(10, 7))
for tolerance_dir in tolerance_dirs:
    #plt.plot(ksp_iterations[tolerance_dir], label='Tolerance: {} (Mean Runtime: {}s)'.format(tolerance_dir, round(mean_runtimes[tolerance_dir], 2)))
    plt.plot(ksp_iterations[tolerance_dir], label=f"{tolerance_dir} (Mean Runtime: {round(mean_runtimes[tolerance_dir], 2)}s)")

plt.xlabel('Policy Iteration')
plt.ylabel('KSP Iterations')
plt.title('KSP Iterations per PI Step for Different Relative Tolerances')
plt.legend()
plt.grid(True)
plt.savefig(f"{path}/SolverTolerance_{os.path.basename(path)}.png", dpi=300)
print(f"Saved figure to {path}/SolverTolerance_{os.path.basename(path)}.png")