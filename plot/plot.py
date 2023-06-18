import json
import matplotlib.pyplot as plt
import numpy as np

# path to your json file
json_file_path = "../cmake-build-debug/out/stats.json"

with open(json_file_path, 'r') as f:
    data = json.load(f)

fig, axs = plt.subplots(2, figsize=(10, 10))

for idx, solver_run in enumerate(data['solver_runs']):
    # Compute cumulative time
    cumulative_time = [sum([item['computation_time'] for item in solver_run[:i+1]]) for i in range(len(solver_run))]

    # Plot residuals vs iterations
    axs[0].semilogy([item['pi_iteration'] for item in solver_run], [item['residual'] for item in solver_run], label=f'Run {idx+1}')
    axs[0].set_xlabel('Iterations')
    axs[0].set_ylabel('Residual $||_\infty$')
    axs[0].legend()

    # Plot residuals vs cumulative time
    axs[1].semilogy(cumulative_time, [item['residual'] for item in solver_run], label=f'Run {idx+1}')
    axs[1].set_xlabel('Cumulative Time [ms]')
    axs[1].set_ylabel('Residual $||_\infty$')
    axs[1].legend()

# Display plot
plt.tight_layout()
plt.savefig('residuals.png')