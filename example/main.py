import madupite as md

with md.PETScContextManager():
    mdp = md.PyMDP()
    mdp.setOption("-mode", "MINCOST")
    mdp.setOption("-discount_factor", "0.9")
    mdp.setOption("-max_iter_pi", "20")
    mdp.setOption("-max_iter_ksp", "1000")
    mdp.setOption("-num_pi_runs", "1")
    mdp.setOption("-rtol_ksp", "1e-4")
    mdp.setOption("-atol_pi", "1e-10")
    mdp.setOption("-file_probabilities", "100_50_0.1/P.bin")
    mdp.setOption("-file_costs", "100_50_0.1/g.bin")
    mdp.setOption("-file_stats", "py_stats.json")
    mdp.setOption("-file_policy", "py_policy.out")
    mdp.setOption("-file_cost", "py_cost.out")
    mdp.setValuesFromOptions()
    mdp.loadFromBinaryFile()
    mdp.inexactPolicyIteration()
