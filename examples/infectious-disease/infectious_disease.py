import matplotlib.colors as mcolors
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import binom  # for binomial transition probabilities

import madupite

# See Gargiani et al. (2024, arXiv:2404.06136), Chapter 5 for a detailed description of the model.


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

    def g(self, state, action):
        a1, a2 = self.a2ij(action)
        cf = self.cf_a1[a1] + self.cf_a2[a2]
        cq = self.cq_a1[a1] * self.cq_a2[a2]
        return (
            self.weights[0] * cf
            - self.weights[1] * cq
            + self.weights[2] * self.ch(state)
        )

    def q(self, state, action):
        a1, a2 = self.a2ij(action)
        beta = 1.0 - 1.0 * state / self.population_size
        return 1 - np.exp(-beta * self.psi[a1] * self.lambda_[a2])

    def P(self, state, action):
        if state == self.population_size:
            return [1.0], [self.population_size]

        q_prob = self.q(state, action)
        ev = state * q_prob
        start = int(max(0, ev - self.num_transitions // 2))
        end = int(min(state, ev + self.num_transitions // 2 - 1))

        binom_values = binom.pmf(np.arange(start, end + 1), state, q_prob)
        sum_binom = np.sum(binom_values)

        vals = []
        indices = []
        for i in range(start, end + 1):
            next_state = self.population_size - i
            prob = binom_values[i - start] / sum_binom
            vals.append(prob)
            indices.append(next_state)

        return vals, indices

    def visualize_transition_probabilities(self):
        cmap_hot_r = plt.get_cmap("hot_r")
        newcolors = cmap_hot_r(np.linspace(0.32, 0.75, 1000))
        white = np.array([1, 1, 1, 1])
        newcolors[0, :] = white
        newcmp = mcolors.ListedColormap(newcolors)

        fig, axs = plt.subplots(4, 5, figsize=(20, 16))

        for action in range(self.num_actions):
            matrix = np.zeros((self.num_states, self.num_states))
            for state in range(self.num_states):
                vals, indices = self.P(state, action)
                matrix[state, indices] = vals

            i, j = self.a2ij(action)
            ax = axs[j, i]
            im = ax.imshow(matrix, cmap=newcmp, interpolation="nearest")
            ax.set_title(f"Action Index: ({i}, {j}) = {action}")

        fig.colorbar(im, ax=axs.ravel().tolist(), shrink=0.6)
        plt.savefig("transition_probabilities.png", dpi=300)

    def visualize_stage_costs(self):
        stage_cost_matrix = np.empty((self.num_states, self.num_actions))
        for state in range(self.num_states):
            for action in range(self.num_actions):
                stage_cost_matrix[state, action] = self.g(state, action)

        fig, ax = plt.subplots(
            figsize=(6, 10)
        )  # Adjusted figure size for vertical orientation
        im = ax.imshow(
            stage_cost_matrix, cmap="hot_r", interpolation="nearest", aspect="auto"
        )
        ax.set_title("Stage Cost Matrix")

        # Set x-axis (action) ticks and label
        ax.set_xticks(range(0, 20, 2))  # Show every other tick to avoid crowding
        ax.set_xticklabels(range(0, 20, 2))
        ax.set_xlabel("Action Index (Higher Index = Stricter Measures)")

        # Set y-axis (state) ticks and label
        num_y_ticks = 5  # Adjust this number for desired density of y-ticks
        y_ticks = np.linspace(0, self.num_states - 1, num_y_ticks, dtype=int)
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_ticks)
        ax.set_ylabel("State Index (Number of Susceptible Individuals)")

        fig.colorbar(im, ax=ax, shrink=0.6)
        plt.tight_layout()  # Adjust layout to prevent clipping of labels
        plt.savefig("stage_costs.png", dpi=300)


# parameter
population_size = 5000
num_transitions = (
    100  # max. non-zeros in transition matrix (to keep it sparse and efficient)
)

psi_hygiene = [0.25, 0.125, 0.08, 0.05, 0.03]
cf_hygiene = [0, 1, 5, 6, 9]
cq_hygiene = [1, 0.7, 0.5, 0.4, 0.05]

lambda_social_distancing = np.array([0.2, 0.16, 0.1, 0.01]) * population_size
cf_social_distancing = [0, 1, 10, 30]
cq_social_distancing = [1, 0.9, 0.5, 0.1]

weights = [5, 20, 0.05]  # [wf, wq, wh]


def main():
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

    rank_id, n_ranks = madupite.mpi_rank_size()
    if rank_id == 0:
        idm.visualize_stage_costs()
        idm.visualize_transition_probabilities()

    prealloc = madupite.MatrixPreallocation()
    prealloc.o_nz = num_transitions
    prealloc.d_nz = num_transitions

    g = madupite.createStageCostMatrix(
        name="g", numStates=idm.num_states, numActions=idm.num_actions, func=idm.g
    )
    P = madupite.createTransitionProbabilityTensor(
        name="P",
        numStates=idm.num_states,
        numActions=idm.num_actions,
        func=idm.P,
        preallocation=prealloc,
    )

    mdp = madupite.MDP()

    mdp.setOption("-mode", "MINCOST")
    mdp.setOption("-discount_factor", "0.99")
    mdp.setOption("-max_iter_pi", "200")
    mdp.setOption("-max_iter_ksp", "1000")
    mdp.setOption("-alpha", "0.1")
    mdp.setOption("-atol_pi", "1e-8")
    mdp.setOption("-overwrite", "true")
    mdp.setOption("-verbose", "true")
    mdp.setOption("-file_stats", "idm_stats.json")
    mdp.setOption("-file_cost", "idm_reward.out")
    mdp.setOption("-file_policy", "idm_policy.out")
    mdp.setOption("-ksp_type", "gmres")

    mdp.setStageCostMatrix(g)
    mdp.setTransitionProbabilityTensor(P)

    mdp.solve()


if __name__ == "__main__":
    main()
