from mdp import *

z_transitionProbabilityMatrix = np.array([[0.975, 0.025], [0.025, 0.975]])
z = np.array([0.726, 1.377])
nz = 2
nk = 20
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
print("Capital stock grid:")
print(k)

def availableResources():
    """precomputation of Matrix B"""
    B = np.zeros((nk, nz))
    for i in range(nk):
        for j in range(nz):
            B[i, j] = z[j] * (k[i]**rho) + k[i]
    return B

B = availableResources()
print("Available resources:")
print(B)

def feasibleActions():
    """
    precomputation of Matrix A: find largest capital stock that can be realized given the current state
    A[i, j] = arg max_a { a | 1 <= a <= nk, z[j]*k[i]**rho + k[i] - k[a] >= 0 }
    """
    A = np.zeros((nk*nz, numActions))
    for i in range(nk):
        for j in range(nz):
            for a in range(numActions):
                if B[i, j] - k[a] > 0:
                    A[ij2s(i,j), a] = a
                else:
                    A[ij2s(i,j), a] = -np.inf

    return np.argmax(A, axis=1)

A = feasibleActions()
print("Feasible actions:")
print(A)

def constructFromPolicy(actionInd):
    P = np.zeros((nk*nz, nk*nz))
    r = np.empty((nk*nz))

    for i in range(nk):
        for j in range(nz):
            s = ij2s(i, j)
            if actionInd <= A[s]: # action must be feasible!
                P[s, actionInd*nz:(actionInd+1)*nz] = z_transitionProbabilityMatrix[j, :]
                #print(f"B[{i}, {j}] = {B[i, j]}; k[{actionInd}] = {k[actionInd]}; gamma = {gamma}")
                r[s] = (B[i, j] - k[actionInd])**gamma / gamma
            else:
                P[s, :] = np.zeros((1, nk*nz))
                r[s] = -np.inf

    return P, r

def generateTransitionTensor():
    P = np.zeros((numActions, nk*nz, nk*nz))
    r = np.zeros((nk*nz, numActions))
    for a in range(numActions):
        P[a, :, :], r[:, a] = constructFromPolicy(a)
    return P, r

P, r = generateTransitionTensor()
mdp = MDP(nk*nz, numActions, beta)
mdp.transitionTensor_ = P
mdp.stageCosts_ = r

V0 = np.zeros((nk*nz, 1))
policy0 = mdp.extractGreedyPolicy(V0)

result, policy = mdp.policyIteration(policy0, V0)

print("Feasible actions:")
print(A)
print("Optimal policy:")
print(policy)
print("Optimal value function:")
print(result[-1][1])

print("Suboptimality gap (inf norm):")
v_opt = np.array(result[-1][1])
for i in range(1, len(result)):
    print(f"{result[i][0]} {np.linalg.norm(np.array(result[i][1]) - v_opt, ord=np.inf)}")
