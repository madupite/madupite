# used to check correctness of C++ parallel implementation.

import sys
sys.path.append("archive/python-numpy-implementation/")
import mdp
import numpy as np

mdp = mdp.MDP(100, 10, 0.9)

# get tensor from csv file data/100_10_0.100000/P.csv
P = np.loadtxt("data/100_10_0.100000/P.csv", delimiter=",")
g = np.loadtxt("data/100_10_0.100000/g.csv", delimiter=",")

# reshape P to 3d tensor
n, m = P.shape
print(n, m)
mdp.transitionTensor_ = np.reshape(P, (100, 10, 100)).transpose((1, 0, 2))
mdp.stageCosts_ = g.copy()

# test greedy policy
V = np.ones((100, 1))
policy = mdp.extractGreedyPolicy(V)

result, policy = mdp.policyIteration(policy, V)
print(policy)

# print optimal cost
print(result[-1][1])