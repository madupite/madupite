from mdp import *
import numpy as np
import sys

def main():

    states = int(sys.argv[1])
    actions = int(sys.argv[2])
    seed = int(sys.argv[3])
    zeroProb = float(sys.argv[4]) # percentage of zero probabilities
    highCostProb = float(sys.argv[5]) # percentage of high costs (10x)

    mdp = MDP(states, actions, 0.1, seed, 1e-14, 300)
    mdp.generateTransitionTensor(zeroProb)
    mdp.generateStageCosts(highCostProb)

    np.save(f"data/transitionTensor_{states}_{actions}_{seed}_{zeroProb}", mdp.transitionTensor_)
    np.save(f"data/stageCosts_{states}_{actions}_{seed}_{highCostProb}", mdp.stageCosts_)



if __name__ == "__main__":
    main()