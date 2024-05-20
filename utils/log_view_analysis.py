import pandas as pd
import matplotlib.pyplot as plt
import re

# read file and store it in data
with open('log_view.txt', 'r') as file:
    data = file.read()

start = data.find("--- Event Stage 0: Main Stage")
start = data.find("\n\n", start) # find the next new line
end = data.find("------------------------------------------------------------------------------------------------------------------------", start)
data = data[start:end]

headers = ["event", "count_max", "count_ratio", "time_max", "time_ratio", "flop_max", "flop_ratio", "flop_mess", "flop_avg_len", "flop_reduct", 
           "global_T", "global_F", "global_M", "global_L", "global_R", "stage_T", "stage_F", "stage_M", "stage_L", "stage_R", "total_mflop_s"]

# split by new line and then by space
lines = data.strip().split('\n')
split_lines = [re.split(r'\s+', line) for line in lines]

# Creating DataFrame
df = pd.DataFrame(split_lines, columns=headers)

# Printing DataFrame
print(df)

# analyze the MDP functions runtime
# MDP::solve calls the other MDP:: functions
# plot runtime of functions as share of MDP::solve runtime
"""
2                       MDP::solve         2         1.0  6.9629e-03        1.0  1.38e+06        1.0  ...      100      65     100       0     100     100           198
3         MDP::constructFromPolicy        10         1.0  6.1627e-04        1.0  0.00e+00        0.0  ...        0       6       0       0       0       0             0
4         MDP::extractGreedyPolicy        12         1.0  2.0799e-03        1.0  9.61e+05        1.0  ...        0      19      70       0       0       0           462
5   MDP::iterativePolicyEvaluation        10         1.0  2.8766e-03        1.0  4.18e+05        1.0  ...        0      27      30       0       0       0           145"""

# extract the data (contain MDP)
mdp_data = df[df['event'].str.contains("MDP")]
