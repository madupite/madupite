import os
import json
import numpy as np
import matplotlib.pyplot as plt
import argparse

# file path as command line argument
parser = argparse.ArgumentParser(description='Plot runtimes of different Krylov algorithms for different problem sizes')
parser.add_argument('--path', required=True, help='path of the output data (parent directory of method and problem size directories)')
args = parser.parse_args()
main_dir = args.path

# Get all first level directories = methods
method_dirs = [d for d in os.listdir(main_dir) if os.path.isdir(os.path.join(main_dir, d))]

# Generate dict with methods and problem sizes as keys and list of runtimes as values
runtime = {}

for method_dir in method_dirs:
    problem_dirs = [d for d in os.listdir(os.path.join(main_dir, method_dir)) if os.path.isdir(os.path.join(main_dir, method_dir, d))]
    problem_sizes = [int(problem_dir)+1 for problem_dir in problem_dirs]

    for problem_dir in problem_dirs:
        json_path = os.path.join(main_dir, method_dir, problem_dir, 'stats.json')
        with open(json_path, 'r') as f:
            data = json.load(f)
        total_runtime_per_run = []
        for run in data['solver_runs']:
            total_runtime = sum([iteration['computation_time'] for iteration in run])
            total_runtime_per_run.append(total_runtime / 1000)

        # If the method is not in the dictionary, add it
        if method_dir not in runtime:
            runtime[method_dir] = {}
        
        runtime[method_dir][problem_dir] = total_runtime_per_run

# Create a subplot
fig, ax1 = plt.subplots()

colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']

# Plot runtime
for i, (method, runtimes) in enumerate(runtime.items()):
    states_counts = []
    means = []
    mins = []
    maxs = []
    for problem_size, runtimes in runtimes.items():
        means.append(np.mean(runtimes))
        mins.append(np.min(runtimes))
        maxs.append(np.max(runtimes))
        states_counts.append(int(problem_size)*2)

    # sort data by states_counts
    states_counts, means, mins, maxs = zip(*sorted(zip(states_counts, means, mins, maxs)))

    ax1.set_xlabel('#states')
    ax1.set_ylabel('Runtime [s]')
    line1 = ax1.plot(states_counts, means, color=colors[i % len(colors)], label=f"{method} mean runtime")
    fill1 = ax1.fill_between(states_counts, mins, maxs, color=colors[i % len(colors)], alpha=0.2)

#grid
ax1.grid(axis='y', linestyle='--')
ranks = data["numRanks"]
plt.title(f"Performance of different Krylov methods for IDM on {ranks} ranks")

# legend
fig.legend(loc='upper left', bbox_to_anchor=(0.1, 0.9), fontsize='small')
fig.tight_layout()
slurm_id = os.path.basename(os.path.normpath(main_dir))
# save to parent directory of main_dir
path = os.path.dirname(main_dir)
plt.savefig(os.path.join(path, f'SolverType_{slurm_id}.png'), dpi=300)
print(f"Plot saved to {os.path.join(path, f'SolverType_{slurm_id}.png')}")
