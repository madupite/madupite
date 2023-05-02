from mdp import *
import sys
import os

def main():

    states = 200
    actions = 20
    discount = 0.9
    sparsity = 0.1
    seed = 8624

    mdp = MDP(states, actions, discount, seed, 1e-14, 300)

    filename_P = f"../../data/P_{states}_{actions}_{sparsity:0.6f}_{seed}.csv"
    filename_g = f"../../data/g_{states}_{actions}_{seed}.csv"

    if not os.path.exists(filename_P) or not os.path.exists(filename_g):
        print("file not found. aborting.")
        sys.exit(1)

    M = np.loadtxt(filename_P, delimiter=",")
    P = np.transpose(M.reshape(states, actions, states), (1, 0, 2)) # reshape from n x (n*m) to m x n x n tensor
    g = np.loadtxt(filename_g, delimiter=",")

    print(P.shape)
    print(g.shape)
    print(g[0,:])
    print(g[3,:])
    print(P.dtype)
    print(g.dtype)

    print(P[1,3,0:20])
    print(P[1,8,0:20])
    #print(P[0,:,:])
    # print row sum of P[0,:,:]
    #print(np.sum(P[0,:,:], axis=1))

    mdp.transitionTensor_ = P.copy()
    mdp.stageCosts_ = g.copy()

    V0 = np.ones(mdp.numStates_)
    policy0 = mdp.extractGreedyPolicy(V0)
    #print(policy0)

    # VI_res, VI_policy = mdp.valueIteration(V0, 10000)
    PI_res, PI_policy = mdp.policyIteration(policy0, V0)
    V_iPI, iPI_policy = mdp.inexactPolicyIteration(V0, 200)
    V_opt = PI_res[-1][1]
    #V_VI = VI_res[-1][1]
    # print(*(np.array(V_opt) - np.array(V_VI)))
    print(*V_iPI)
    print(*iPI_policy)

    #print((PI_policy == iPI_policy))

if __name__ == "__main__":
    main()