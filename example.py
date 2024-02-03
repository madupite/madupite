import madupite as md

with md.PETScContextManager():
    mdp = md.PyMDP()
    
    mdp.setValuesFromFile("./petsc_options.txt")

    mdp.loadFromBinaryFile()

    mdp.inexactPolicyIteration()