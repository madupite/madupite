import os
import json
import numpy as np
import matplotlib.pyplot as plt
import argparse

# file path as command line argument
parser = argparse.ArgumentParser(description='Plot runtimes of iPI for different problem sizes')
parser.add_argument('--path', required=True, help='path of the output data (parent directory of problem size directories)')
args = parser.parse_args()
main_dir = args.path

# Get all directories = problemSizes
problem_dirs = [d for d in os.listdir(main_dir) if os.path.isdir(os.path.join(main_dir, d))]
problem_sizes = [int(problem_dir)*2 for problem_dir in problem_dirs]  # dir = nk; #states = nk*nz (nz=2)

# lists to store data
states_counts = []
means = []
mins = []
maxs = []

# Generate dict with problem sizes as keys and list of runtimes as values
runtime = {}

for problem_dir in problem_dirs:
    json_path = os.path.join(main_dir, problem_dir, 'stats.json')
    with open(json_path, 'r') as f:
        data = json.load(f)
    total_runtime_per_run = []
    for run in data['solver_runs']:
        total_runtime = sum([iteration['computation_time'] for iteration in run])
        total_runtime_per_run.append(total_runtime / 1000)

    runtime[problem_dir] = total_runtime_per_run

# make runtime vs problem size plot
# calculate mean, min, max

for problem_size, runtimes in runtime.items():
    means.append(np.mean(runtimes))
    mins.append(np.min(runtimes))
    maxs.append(np.max(runtimes))
    states_counts.append(int(problem_size)*2)  # double problem size is #states

# sort data by states_counts
states_counts, means, mins, maxs = zip(*sorted(zip(states_counts, means, mins, maxs)))

# Create a subplot
fig, ax1 = plt.subplots()

# Plot runtime
color = 'tab:blue'
ax1.set_xlabel('#states')
ax1.set_ylabel('Runtime [s]', color=color)
line1 = ax1.plot(states_counts, means, color=color, label="mean runtime")
fill1 = ax1.fill_between(states_counts, mins, maxs, color=color, alpha=0.2)
ax1.tick_params(axis='y', labelcolor=color)
#grid
ax1.grid(axis='y', linestyle='--')

plt.title(f"Runtime vs problem size (#states) for iPI")

# legend
fig.legend(loc='upper left', bbox_to_anchor=(0.1, 0.9), fontsize='small')
fig.tight_layout()
slurm_id = os.path.basename(os.path.normpath(main_dir))
# save to parent directory of main_dir
path = os.path.dirname(main_dir)
plt.savefig(os.path.join(path, f'runtime_vs_problem_size_{slurm_id}.png'), dpi=300)
print(f"Plot saved to {os.path.join(path, f'runtime_vs_problem_size_{slurm_id}.png')}")
