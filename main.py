from mdp import *
from plot import *


def main():

    # Bertsekas example
    # mdp = MDP(2, 2, 0.9, 42, 1e-14, 1)
    # mdp.stageCosts_ = np.array([[2, 0.5], [1, 3]])
    # mdp.transitionTensor_ = np.array([[[0.75, 0.25], [0.75, 0.25]], [[0.25, 0.75], [0.25, 0.75]]])

    mdp = MDP(3000, 50, 0.6, 42, 1e-14, 300)
    mdp.generateTransitionTensor(0.2)
    mdp.generateStageCosts(0.4)

    V0 = np.zeros(mdp.numStates_)
    policy0 = mdp.extractGreedyPolicy(V0)

    VI_res, VI_policy = mdp.valueIteration(V0, 1000)
    PI_res, PI_policy = mdp.policyIteration(policy0, V0)
    OPI_res, OPI_policy = mdp.optimisticPolicyIteration(policy0, V0, 20)

    plotResult(mdp, VI_res, PI_res, OPI_res)

if __name__ == "__main__":
    main()