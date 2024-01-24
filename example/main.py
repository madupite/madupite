import madupite as md

with md.PETScContextManager():
    mdp = md.PyMDP()
    mdp.setOption("-mode", "MINCOST")
    mdp.setOption("-discountFactor", "0.9")
    mdp.setOption("-maxIter_PI", "20")
    mdp.setOption("-maxIter_KSP", "1000")
    mdp.setOption("-numPIRuns", "1")
    mdp.setOption("-rtol_KSP", "1e-4")
    mdp.setOption("-atol_PI", "1e-10")
    mdp.setOption("-file_P", "100_50_0.1/P.bin")
    mdp.setOption("-file_g", "100_50_0.1/g.bin")
    mdp.setOption("-file_stats", "py_stats.json")
    mdp.setOption("-file_policy", "py_policy.out")
    mdp.setOption("-file_cost", "py_cost.out")
    mdp.setValuesFromOptions()
    mdp.loadFromBinaryFile()
    mdp.inexactPolicyIteration()
