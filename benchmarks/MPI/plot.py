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



config_dirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
runtimes = {}
mean_runtimes = {}
min_runtimes = {}
max_runtimes = {}

for config_dir in config_dirs:
    runtimes[config_dir] = []
    mean_runtimes[config_dir] = []
    min_runtimes[config_dir] = []
    max_runtimes[config_dir] = []
    json_path = os.path.join(path, config_dir, 'stats.json')
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # compute mean runtime
    for run in data['solver_runs']:
        total_runtime = sum([iteration['computation_time'] for iteration in run])
        runtimes[config_dir].append(total_runtime / 1000)
    mean_runtimes[config_dir] = np.mean(runtimes[config_dir])
    min_runtimes[config_dir] = np.min(runtimes[config_dir])
    max_runtimes[config_dir] = np.max(runtimes[config_dir])



# boxplot runtimes

plt.figure(figsize=(10, 7))
plt.boxplot(runtimes.values(), labels=runtimes.keys())
plt.xlabel('MPI Configuration')
plt.ylabel('Runtime [s]')
plt.title('Runtime for Different MPI Configurations')
plt.grid(True)
plt.savefig(f"{path}/MPIConfigurations_{os.path.basename(path)}.png", dpi=300)
print(f"Saved figure to {path}/MPIConfigurations_{os.path.basename(path)}.png")

# cloud/density plot runtimes
# plt.figure(figsize=(10, 7))
# plt.plot(runtimes.keys(), runtimes.values(), 'o')
# plt.xlabel('MPI Configuration')
# plt.ylabel('Runtime [s]')
# plt.title('Runtime for Different MPI Configurations')
# plt.grid(True)
# plt.savefig(f"{path}/MPIConfigurations_{os.path.basename(path)}.png", dpi=300)
# print(f"Saved figure to {path}/MPIConfigurations_{os.path.basename(path)}.png")
