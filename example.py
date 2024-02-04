import madupite as md
import time

with md.PETScContextManager():
    mdp = md.PyMDP()

    mdp.setValuesFromFile("./petsc_options.txt")

    mdp.loadFromBinaryFile()

    mdp.inexactPolicyIteration()

    def transprob(x, y):
        return [[0, 1, 2], [0.1, 0.2, 0.7]]

    def stagecost(x, y):
        return x * y

    if md.MPI_master():
        start = time.time()
    mdp.createTransitionProbabilities(10000, 500, transprob)
    mdp.createStageCosts(10000, 500, stagecost)
    if md.MPI_master():
        print(time.time() - start)
    mdp.inexactPolicyIteration()
