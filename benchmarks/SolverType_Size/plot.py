import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from plot_util import *
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(description='Plot runtimes of different Krylov algorithms for different problem sizes')
parser.add_argument('--path', required=True, help='path of the output data (parent directory of method and problem size directories)')
args = parser.parse_args()
main_dir = args.path
slurm_id = os.path.basename(main_dir)

# compute runtimes
runtimes = extract_runtimes(main_dir)

# Create a subplot
fig, ax1 = plt.subplots()

colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']

# Plot runtime
for i, (method, variations) in enumerate(runtimes.items()):
    variation_counts = []
    means = []
    mins = []
    maxs = []
    for variation, stats in variations.items():
        means.append(stats['mean'])
        mins.append(stats['min'])
        maxs.append(stats['max'])
        variation_counts.append(variation)

    # sort data by variation_counts
    variation_counts, means, mins, maxs = zip(*sorted(zip(variation_counts, means, mins, maxs)))

    ax1.set_xlabel('states')
    ax1.set_ylabel('Runtime [s]')
    line1 = ax1.plot(variation_counts, means, color=colors[i % len(colors)], label=f"{method}")
    fill1 = ax1.fill_between(variation_counts, mins, maxs, color=colors[i % len(colors)], alpha=0.2)

#grid
ax1.grid(axis='y', linestyle='--')
plt.title("Performance of different Krylov methods for different variations")

# legend
fig.legend(loc='upper left', bbox_to_anchor=(0.1, 0.9), fontsize='small')
fig.tight_layout()

plt.savefig(f"{main_dir}/SolverType_Size_{slurm_id}.png", dpi=300)
print(f"Plot saved to {main_dir}/SolverType_Size_{slurm_id}.png")
plt.show()
