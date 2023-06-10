import sys
sys.path.append("../archive/python-numpy-implementation/")
from mdp import *


z_transitionProbabilityMatrix = np.array([[0.975, 0.025], [0.025, 0.975]])
z = np.array([0.726, 1.377])
nz = 2
nk = 500
numActions = nk

gamma = 0.5 # risk aversion parameter, could also be -5.0 (high risk aversion)
beta = 0.98 # discount factor
rho = 0.33 # f(k) = k^rho (implied production function)

def ij2s(i, j):
    return i * nz + j # row major

def s2ij(s):
    return s // nz, s % nz

def generate_k_interval():
    k_star_z1 = (beta * rho * z[0] / (1 - beta)) ** (1 / (1 - rho))
    k_star_z2 = (beta * rho * z[1] / (1 - beta)) ** (1 / (1 - rho))
    # generate linspaced interval st. 10% of the values in the interval are below k_star_z1 and 10% are above k_star_z2
    min_k = k_star_z1 - (0.1 * (k_star_z2 - k_star_z1))
    max_k = k_star_z2 + (0.1 * (k_star_z2 - k_star_z1))
    return np.linspace(min_k, max_k, nk)

k = generate_k_interval()

def availableResources():
    """precomputation of Matrix B"""
    B = np.zeros((nk, nz))
    for i in range(nk):
        for j in range(nz):
            B[i, j] = z[j] * (k[i]**rho) + k[i]
    return B

B = availableResources()

def feasibleActions():
    """
    precomputation of Matrix A: find largest capital stock that can be realized given the current state
    A[i, j] = arg max_a { a | 1 <= a <= nk, z[j]*k[i]**rho + k[i] - k[a] >= 0 }
    """
    A = np.zeros((nk*nz, numActions))
    for i in range(nk):
        for j in range(nz):
            for a in range(numActions):
                if B[i, j] - k[a] >= 0:
                    A[ij2s(i,j), a] = a
                else:
                    A[ij2s(i,j), a] = -np.inf

    return np.argmax(A, axis=1)

A = feasibleActions()




            



