import time

import madupite as md


def transprob(x, y):
    return [[0, 1, 2], [0.1, 0.2, 0.7]]


def stagecost(x, y):
    return x * y


with md.PETScContextManager():
    mdp = md.PyMDP()

    # solve MDP from file
    mdp.setValuesFromFile("./petsc_options.txt")
    mdp.loadFromBinaryFile()
    mdp.inexactPolicyIteration()

    start = time.time() if md.MPI_master() else 0

    # create MDP using functions
    mdp.setOption("-num_states", "10000")
    mdp.setOption("-num_actions", "500")
    mdp.setValuesFromOptions()
    mdp.createTransitionProbabilities(transprob)
    mdp.createStageCosts(stagecost)

    # change inner solver and discount factor
    mdp.setOption("-ksp_type", "tfqmr")
    mdp["-discount_factor"] = "0.999"
    mdp.setValuesFromOptions()
    end = time.time() if md.MPI_master() else 0

    mdp.inexactPolicyIteration()
    print("Time to generate MDP: ", end - start)
