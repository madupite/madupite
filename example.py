import madupite as md

with md.PETScContextManager():
    mdp = md.PyMDP()

    mdp.setValuesFromFile("./petsc_options.txt")

    mdp.loadFromBinaryFile()

    mdp.inexactPolicyIteration()

    def transprob(x, y):
        return [[0,1,2], [0.1, 0.2, 0.7]]
    
    def stagecost(x, y):
        return x*y

    mdp.createTransitionProbabilities(100, 50, transprob)
    mdp.createStageCosts(100, 50, stagecost)
    mdp.inexactPolicyIteration()
