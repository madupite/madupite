import os
import json
import numpy as np

def extract_runtimes(path):
    # Get all first level directories = methods
    method_dirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]

    # Generate dict with methods and variations as keys and runtime stats as values
    runtimes = {}

    for method_dir in method_dirs:
        variation_dirs = [d for d in os.listdir(os.path.join(path, method_dir)) if os.path.isdir(os.path.join(path, method_dir, d))]

        for variation_dir in variation_dirs:
            json_path = os.path.join(path, method_dir, variation_dir, 'stats.json')
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            total_runtimes = []
            for run in data['solver_runs']:
                total_runtime = sum([iteration['computation_time'] for iteration in run])
                total_runtimes.append(total_runtime / 1000)  # convert to seconds

            # Calculate mean, min and max
            mean_runtime = np.mean(total_runtimes)
            min_runtime = np.min(total_runtimes)
            max_runtime = np.max(total_runtimes)

            # If the method is not in the dictionary, add it
            if method_dir not in runtimes:
                runtimes[method_dir] = {}
            
            # Add variation and its runtime stats to the method
            runtimes[method_dir][float(variation_dir)] = {
                'mean': mean_runtime,
                'min': min_runtime,
                'max': max_runtime,
            }

    return runtimes
