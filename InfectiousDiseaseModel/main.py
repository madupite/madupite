import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom
import sys
sys.path.append("../archive/python-numpy-implementation/")
from mdp import *

# SIS model for infectious diseases

# parameters
N = 100 # population size
discountFactor = 0.7 # no clue, nothing in the paper?
numStates = N + 1
numA1 = 5 # hygienic measures
numA2 = 4 # social distancing
numActions = 5*4 # by design of model

# disease parameters
#r0 = 0.25 #0.115 # never used!
lambda0 = 80


# generate stage costs
# a1: hygienic measures [0, 4]
# a2: social distancing [0, 3]
#ij2a = lambda i, j: i * numA1 + j # action index mapping
def ij2a(i, j):
    return j * numA1 + i

#a2ij = lambda a: (a // numA1, a % numA1) # action index mapping
def a2ij(a):
    return (a % numA1, a // numA1)

# financial costs (cf) [dependent on action]
cf_a1 = [0, 1, 5, 6, 9]
cf_a2 = [0, 1, 10, 30]

# quality of life costs (cq) [dependent on action]
cq_a1 = [1, 0.7, 0.5, 0.4, 0.05]
cq_a2 = [1, 0.9, 0.5, 0.1]

# health costs (ch) [dependent on state]
#ch = lambda s: (N - s)**1.1
def ch(s):
    return (N - s)**1.1

# total costs (ct) [dependent on action and state]
wf = 2
wq = 10
wh = 0.5
def g(s, a):
    a1 = a2ij(a)[0]
    a2 = a2ij(a)[1]
    return wf * (cf_a1[a1] + cf_a2[a2]) - wq * (cq_a1[a1] * cq_a2[a2]) + wh * ch(s)


# generate numStates x numActions stage cost matrix
stageCostMatrix = np.zeros((numStates, numActions))
for s in range(numStates):
    for a in range(numActions):
        stageCostMatrix[s, a] = g(s, a)

# print(stageCostMatrix)
# store as csv
np.savetxt("stageCostMatrix.csv", stageCostMatrix, delimiter=",", fmt='%1.2f')

# create matplotlib heatmap of stage costs (higher value red, lower green)
plt.imshow(stageCostMatrix, cmap='hot', interpolation='nearest')
plt.colorbar()
plt.savefig("stageCostMatrix.png")


# generate transition probabilities
# s = N (0 infected) is absorbing state

#r_ = [0.115, 0.0575, 0.03, 0.02, 0.012]
r_ = [0.25, 0.125, 0.08, 0.05, 0.03]
#lambda_ = [lambda0, 0.8*lambda0, 0.5*lambda0, 0.05*lambda0]
lambda_ = [20, 16, 10, 1]
def q(s, a):
    a1 = a2ij(a)[0]
    a2 = a2ij(a)[1]
    beta = 1.0 - s / N #1 / (N - s) # probability of a contact being with an infected person
    return 1 - np.exp(-beta * r_[a1] * lambda_[a2])

# generate numActions x numStates x numStates transition probability matrix
transitionProbabilityTensor = np.zeros((numActions, numStates, numStates))
# P(
max_err = 0

for a in range(numActions):
    for s in range(numStates - 1):
        for i in range(numStates):
            #s_ = s - i
            #s_ = i
            s_ = N - i
            if i > s:
                transitionProbabilityTensor[a, s, s_] = 0
            else:
                transitionProbabilityTensor[a, s, s_] = binom.pmf(i, s, q(s, a))

        max_err = max(np.sum(transitionProbabilityTensor[a, s, :]) - 1, max_err)
    transitionProbabilityTensor[a, numStates - 1, numStates - 1] = 1

"""
for a in range(numActions):
    for s in range(numStates - 1):
        for x in range(numStates):
            if N - s <= x and x <= N:
                transitionProbabilityTensor[a, s, x] = binom.pmf(N - x, s, q(s, a))
            else:
                transitionProbabilityTensor[a, s, x] = 0
        max_err = max(np.sum(transitionProbabilityTensor[a, s, :]) - 1, max_err)
    transitionProbabilityTensor[a, numStates - 1, numStates - 1] = 1
"""
#np.savetxt("transitionProbabilityTensor.csv", transitionProbabilityTensor[0, :, :], delimiter=",", fmt='%1.2f')

"""
# c++ version: P is numStates*numActions x numStates matrix
transitionProbabilityTensor = np.zeros((numStates * numActions, numStates))

for s in range(numStates):
    for a in range(numActions):
        if s == N:
            transitionProbabilityTensor[s * numActions + a, s] = 1
            continue
        print(f"  q({s}, {a}) = {q(s, a)}")
        for i in range(s+1):
            s_ = N - i
            transitionProbabilityTensor[s * numActions + a, s_] = binom.pmf(i, s, q(s, a))


P = np.zeros((numStates, numStates))
for s in range(numStates):
    P[s, :] = transitionProbabilityTensor[s * numActions, :]
np.savetxt("P.csv", P, delimiter=",")
"""

#print(transitionProbabilityTensor)
print("max row stochasticity error: ", max_err)

# plot transition probability matrix for all actions (4, 5) subplots
fig, axs = plt.subplots(4, 5, figsize=(20, 20))
for a in range(numActions):
    ax = axs[a // numA1, a % numA1]
    ax.set_title(f"(a1, a2) = {a2ij(a)}")
    im = ax.imshow(transitionProbabilityTensor[a, :, :], cmap='hot', interpolation='nearest')
    ax.set_xlabel("S'")
    ax.set_ylabel("S")

fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
fig.colorbar(im, cax=cbar_ax)
plt.savefig("transitionProbabilityTensor.png")


### --------- MDP --------- ###
mdp = MDP(numStates, numActions, discountFactor)
mdp.transitionTensor_ = transitionProbabilityTensor.copy()
mdp.stageCosts_ = stageCostMatrix.copy()
print(mdp.transitionTensor_.shape)
print(mdp.stageCosts_.shape)

V0 = np.ones(numStates)
policy0 = mdp.extractGreedyPolicy(V0)
print("policy0: ", policy0)

result, policy = mdp.policyIteration(policy0, V0)
print("Optimal policy:")
print(policy)
print("Optimal value function:")
print(result[-1][1])

print("Suboptimality gap (inf norm):")
v_opt = np.array(result[-1][1])
for i in range(1, len(result)):
    print(f"{result[i][0]} {np.linalg.norm(np.array(result[i][1]) - v_opt, ord=np.inf)}")


# print actions and index mapping
print("action index mapping:")
for a in range(numActions):
    print(f"a = {a} -> {a2ij(a)}")



