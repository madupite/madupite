import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json
import argparse
import os

# replace this with your directory
# file path as command line argument
parser = argparse.ArgumentParser(description='Plot policy function of GrowthModel')
parser.add_argument('--path', required=True, help='path of the output data')
args = parser.parse_args()
main_dir = args.path
json_path = os.path.join(main_dir, 'stats.json')
# Assuming the following values exist:
# k_values: 1D array of size nk with possible capital stock values
# z_values: 1D array of size nz with possible technology shock values
# policy: 1D array of size nk*nz with policy values for each state

k_values = np.loadtxt(os.path.join(main_dir, "k.out"), dtype=float)
print(k_values.shape)
z_values = np.array([0.726, 1.377])
print(z_values.shape)
policy = np.loadtxt(os.path.join(main_dir, "policy.out"))
print(policy.shape)
nk = len(k_values)
nz = len(z_values)

# Initialize 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Create meshgrid for capital stocks and technology shocks
K, Z = np.meshgrid(k_values, z_values)

# Reshape policy array to 2D
Policy = policy.reshape(nz, nk)  # Changed order


# Plot surface
ax.plot_surface(K, Z, Policy, cmap='viridis')

# Set labels
ax.set_xlabel('Capital Stock')
ax.set_ylabel('Technology Shock')
ax.set_zlabel('Optimal Capital Stock')
ax.set_yticks(z_values)
# Show plot
plt.savefig(os.path.join(main_dir, "policy.png"), dpi=300)
plt.show()
