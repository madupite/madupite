import numpy as np
from scipy.stats import binom

import madupite as md


class InfectiousDiseaseModel:
    def __init__(
        self,
        population_size,
        num_transitions,
        cf_a1,
        cf_a2,
        cq_a1,
        cq_a2,
        psi,
        lambda_,
        weights,
    ):
        self.population_size = population_size
        self.num_states = (
            population_size + 1
        )  # including state of 0 susceptibles; state = #susceptibles = population_size - #infected
        self.num_actions = 5 * 4  # 5 hygiene levels, 4 social distancing levels
        self.num_transitions = num_transitions  # number of transitions to consider for each action (such that matrix remains sparse and doesn't have lots of entries in the range of 1e-8)
        self.cf_a1 = cf_a1  # financial cost of hygiene measures
        self.cf_a2 = cf_a2  # financial cost of social distancing measures
        self.cq_a1 = cq_a1  # quality of life impact of hygiene measures (1 = full quality of life, 0 = no quality of life)
        self.cq_a2 = cq_a2  # quality of life impact of social distancing measures (1 = full quality of life, 0 = no quality of life)
        self.psi = psi  # probability that a susceptible person gets infected upon contact with an infected person
        self.lambda_ = lambda_  # contact rate per day
        self.weights = weights  # weights [wf, wq, wh] for cost function (cf, cq, ch)

    def a2ij(
        self, a
    ):  # 1d index to 2d index (i = a1 hygiene, j = a2 social distancing)
        return a % 5, a // 5

    def ch(
        self, state
    ):  # cost of hospitalization (superlinear function of #infected people)
        return (self.population_size - state) ** 1.1

    def g(self, state, action):  # final stage cost
        a1, a2 = self.a2ij(action)
        cf = self.cf_a1[a1] + self.cf_a2[a2]
        cq = self.cq_a1[a1] * self.cq_a2[a2]
        return (
            self.weights[0] * cf
            - self.weights[1] * cq
            + self.weights[2] * self.ch(state)
        )

    def q(
        self, state, action
    ):  # overall probability that a susceptible person becomes infected
        a1, a2 = self.a2ij(action)
        beta = (
            1.0 - 1.0 * state / self.population_size
        )  # beta(s) = probability that the next interaction of a random susceptible person is with an infectious person
        return 1 - np.exp(-beta * self.psi[a1] * self.lambda_[a2])

    def P(self, state, action):
        """
        returns [[<list of next states>], [<list of probabilities>]]
        """
        # hardcode computation, don't use transition_probability_matrices
        a1, a2 = self.a2ij(action)
        q_prob = self.q(state, action)
        ev = state * q_prob
        start = int(max(0, ev - self.num_transitions // 2))
        end = int(min(state, ev + self.num_transitions // 2))
        binom_values = binom.pmf(np.arange(start, end + 1), state, q_prob)
        sum_binom = np.sum(binom_values)
        next_states = [self.population_size - i for i in range(start, end + 1)]
        probabilities = binom_values / sum_binom
        return [next_states, probabilities.tolist()]


# parameter
population_size = 999
num_transitions = (
    100  # max. number of non-zero entries in transition probability matrix per row
)

psi_hygiene = [0.25, 0.125, 0.08, 0.05, 0.03]
cf_hygiene = [0, 1, 5, 6, 9]
cq_hygiene = [1, 0.7, 0.5, 0.4, 0.05]

lambda_social_distancing = np.array([0.2, 0.16, 0.1, 0.01]) * population_size
cf_social_distancing = [0, 1, 10, 30]
cq_social_distancing = [1, 0.9, 0.5, 0.1]

weights = [5, 20, 0.05]  # [wf, wq, wh]

# create model
idm = InfectiousDiseaseModel(
    population_size,
    num_transitions,
    cf_hygiene,
    cf_social_distancing,
    cq_hygiene,
    cq_social_distancing,
    psi_hygiene,
    lambda_social_distancing,
    weights,
)

with md.PETScContextManager():
    mdp = md.PyMDP()
    # mdp.setOption("-ksp_monitor", "") # classic PETSc options are supported as well!
    # mdp.setValuesFromFile("idm_options.txt")
    mdp.setOption("-ksp_type", "tfqmr")
    mdp.setOption("-mat_type", "mpiaij")
    mdp.setOption("-pc_type", "none")
    mdp.setOption("-discount_factor", "0.95")
    mdp.setOption("-mode", "MINCOST")
    mdp.setOption("-max_iter_ksp", "200")
    mdp.setOption("-max_iter_pi", "20")
    mdp.setOption("-rtol_ksp", "1e-4")
    mdp.setOption("-atol_pi", "1e-10")
    mdp.setOption("-file_stats", "idm_stats.out")
    mdp.setOption("-file_policy", "idm_policy.out")
    mdp.setOption("-file_cost", "idm_cost.out")
    mdp.setOption("-num_states", str(idm.num_states))
    mdp.setOption("-num_actions", str(idm.num_actions))
    mdp.setValuesFromOptions()

    mdp.createTransitionProbabilities(idm.P)
    mdp.createStageCosts(idm.g)

    mdp.inexactPolicyIteration()
