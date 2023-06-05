import json
import matplotlib.pyplot as plt
import numpy as np

# path to your json file
json_file_path = "../cmake-build-debug/out/stats.json"

# Load JSON file
with open(json_file_path, 'r') as f:
    data = json.load(f)

# Extract data
computation_time = [d['computation_time'] for d in data['data']]
ksp_iterations = [d['ksp_iterations'] for d in data['data']]
pi_iterations = [d['pi_iteration'] for d in data['data']]
residual = [d['residual'] for d in data['data']]


# Calculate cumulative time
cumulative_time = np.cumsum(computation_time)

# Create a subplot
fig, axs = plt.subplots(2, figsize=(10, 10))

# First subplot
axs[0].semilogy(pi_iterations, residual, 'o-')
axs[0].set_title('Residual Norm vs Iterations')
axs[0].set_xlabel('Iterations')
axs[0].set_ylabel('Residual Norm (log scale)')

# Second subplot
axs[1].semilogy(cumulative_time, residual, 'o-')
axs[1].set_title('Residual Norm vs Cumulative Computation Time')
axs[1].set_xlabel('Cumulative Computation Time')
axs[1].set_ylabel('Residual Norm (log scale)')

# Display plot
plt.tight_layout()
plt.show()