from mdp import *
from plot import *
import sys
import os


def main():

    if len(sys.argv) != 7:
        print("Usage: python main.py <states> <actions> <discount> <seed> <zeroProb> <highCostProb>")
        print("Example: python main.py 2 2 0.9 42 0.1 0.1\n\n")
        sys.exit(1)


    # Bertsekas example
    # mdp = MDP(2, 2, 0.9, 42, 1e-14, 1)
    # mdp.stageCosts_ = np.array([[2, 0.5], [1, 3]])
    # mdp.transitionTensor_ = np.array([[[0.75, 0.25], [0.75, 0.25]], [[0.25, 0.75], [0.25, 0.75]]])
    states = int(sys.argv[1])
    actions = int(sys.argv[2])
    discount = float(sys.argv[3])
    seed = int(sys.argv[4])
    zeroProb = float(sys.argv[5]) # percentage of zero probabilities
    highCostProb = float(sys.argv[6]) # percentage of high costs (10x)
    
    mdp = MDP(states, actions, discount, seed, 1e-14, 300)
    
    # check if directory data exists, if not, create it
    if not os.path.exists("data"):
        os.makedirs("data")
    
    # check if following files exist:
    # data/transitionTensor_{states}_{actions}_{seed}_{zeroProb}.npy
    # data/stageCosts_{states}_{actions}_{seed}_{highCostProb}.npy
    # if so, load them, else generate them
    try:
        print("Loading arrays")
        mdp.transitionTensor_ = np.load(f"data/transitionTensor_{states}_{actions}_{seed}_{zeroProb}.npy")
        mdp.stageCosts_ = np.load(f"data/stageCosts_{states}_{actions}_{seed}_{highCostProb}.npy")
    except:
        print("Failed to load arrays, generating them")
        mdp.generateTransitionTensor(zeroProb)
        mdp.generateStageCosts(highCostProb)
        np.save(f"data/transitionTensor_{states}_{actions}_{seed}_{zeroProb}", mdp.transitionTensor_)
        np.save(f"data/stageCosts_{states}_{actions}_{seed}_{highCostProb}", mdp.stageCosts_)

    V0 = np.zeros(mdp.numStates_)
    policy0 = mdp.extractGreedyPolicy(V0)

    VI_res, VI_policy = mdp.valueIteration(V0, 10000)
    PI_res, PI_policy = mdp.policyIteration(policy0, V0)

    innerIterations = [5, 10, 20, 50]
    OPI_res = []
    for k in innerIterations:
        result, OPI_policy = mdp.optimisticPolicyIteration(policy0, V0, k)
        OPI_res.append([result, k])

    plotResult(mdp, VI_res, PI_res, OPI_res)

if __name__ == "__main__":
    main()