import os
import json
import numpy as np
import matplotlib.pyplot as plt
import argparse

# file path as command line argument
parser = argparse.ArgumentParser(description='Plot runtimes of different Krylov algorithms')
parser.add_argument('--path', required=True, help='path of the output data (parent directory of method directories)')
args = parser.parse_args()
main_dir = args.path

# Get all directories = methods
method_dirs = [d for d in os.listdir(main_dir) if os.path.isdir(os.path.join(main_dir, d))]

# lists to store data
means = {}
mins = {}
maxs = {}

# Generate dict with methods as keys and list of runtimes as values
runtime = {}

states = 0
ranks = 0

for method_dir in method_dirs:
    json_path = os.path.join(main_dir, method_dir, 'stats.json')
    with open(json_path, 'r') as f:
        data = json.load(f)
    total_runtime_per_run = []
    for run in data['solver_runs']:
        total_runtime = sum([iteration['computation_time'] for iteration in run])
        total_runtime_per_run.append(total_runtime / 1000)

    runtime[method_dir] = total_runtime_per_run
    states = data['numStates']
    ranks = data['numRanks']

# make runtime distribution plot
# calculate mean, min, max

for method, runtimes in runtime.items():
    means[method] = np.mean(runtimes)
    mins[method] = np.min(runtimes)
    maxs[method] = np.max(runtimes)

# Create a subplot
fig, ax1 = plt.subplots()

# Plot runtime
ax1.set_ylabel('Runtime [s]')
ax1.boxplot(runtime.values(), labels=runtime.keys())
#grid
ax1.grid(axis='y', linestyle='--')
ax1.set_ylim(0, 2) # cuts off outliers

plt.title(f"Runtime distribution for different Krylov methods (n={states}, {ranks} ranks)")

fig.tight_layout()
slurm_id = os.path.basename(os.path.normpath(main_dir))
# save to parent directory of main_dir
path = os.path.dirname(main_dir)
plt.savefig(os.path.join(path, f'runtime_distribution_{slurm_id}.png'), dpi=300)
print(f"Plot saved to {os.path.join(path, f'runtime_distribution_{slurm_id}.png')}")