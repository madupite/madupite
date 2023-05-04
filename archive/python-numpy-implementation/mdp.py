import numpy as np
import time


class MDP:
    """
    class for a Markov Decision Process
    """

    def __init__(self, numStates, numActions, discount, seed = 42, tol = 1e-14, printInterval = 50):
        self.numStates_ = numStates
        self.numActions_ = numActions
        self.discount_ = discount
        self.seed_ = seed
        self.tol_ = tol
        self.transitionTensor_ = np.empty((numActions, numStates, numStates))
        self.stageCosts_ = np.empty((numStates, numActions))
        self.printInterval_ = printInterval

    def generateTransitionTensor(self, p):
        """
        generates a random transition tensor
        :param p: probability of a transition being zero
        """
        print("Generating transition matrices")

        np.random.seed(self.seed_)
        for i in range(self.numActions_):
            for j in range(self.numStates_):
                for k in range(self.numStates_):
                    rand = np.random.rand()
                    if np.random.rand() < p:
                        rand = 0
                    self.transitionTensor_[i, j, k] = rand
                self.transitionTensor_[i, j, :] /= np.sum(self.transitionTensor_[i, j, :])

    def generateStageCosts(self, p):
        """
        generates random stage costs
        :param p: probability of a stage cost being 10x higher (perturbs uniformity of stage costs)
        """
        print("Generating stage costs")

        np.random.seed(self.seed_)
        self.stageCosts_ = np.random.rand(self.numStates_, self.numActions_) * 10

        # perturb uniformity of stage costs
        for i in range(self.numStates_):
            for j in range(self.numActions_):
                if np.random.rand() < p:
                    self.stageCosts_[i, j] *= 10


    def valueIteration(self, V0, maxIter):

        V = V0.copy()
        V_old = np.zeros(self.numStates_)
        costs = np.empty((self.numStates_, self.numActions_))
        policy = np.zeros(self.numStates_, dtype=int)
        result = [[0, V0.copy().tolist(), time.time()]]

        print("Value Iteration")
        for i in range(1, maxIter):
            # if i % self.printInterval_ == 0:
            #     print("Iteration: ", i)
            
            # calculate costs for all actions
            for state in range(self.numStates_):
                for action in range(self.numActions_):
                    costs[state, action] = self.stageCosts_[state, action] + self.discount_ * np.dot(self.transitionTensor_[action, state, :], V)
            
            # extract optimal cost and policy
            policy[:] = np.argmin(costs, axis=1)
            V_old[:] = V
            V[:] = np.min(costs, axis=1)
 
            # store iteration
            result.append([i, V.copy().tolist(), time.time()])

            # if i % self.printInterval_ == 0:
            #     print(f"{V} | {policy}")

            if np.linalg.norm(V - V_old, ord=np.inf) < self.tol_:
                print("Converged after ", i, " iterations")
                return result, policy

        print("Did not converge after ", maxIter, " iterations")
        return np.zeros(self.numStates_), np.zeros(self.numStates_, dtype=int)

    def extractGreedyPolicy(self, V):
        costs = np.empty((self.numStates_, self.numActions_))
        policy = np.zeros(self.numStates_, dtype=int)
        for stateIndex in range(self.numStates_):
            for actionIndex in range(self.numActions_):
                costs[stateIndex, actionIndex] = self.stageCosts_[stateIndex, actionIndex] + self.discount_ * np.dot(self.transitionTensor_[actionIndex, stateIndex, :], V)
        policy[:] = np.argmin(costs, axis=1)
        return policy

    def extractGreedyPolicy2(self, V):

        policy = np.empty(self.numStates_, dtype=int)
        minCosts = np.full(self.numStates_, np.inf)

        for actionInd in range(self.numActions_):
            currPolicy = np.full(self.numStates_, actionInd, dtype=int)
            g, P = self.constructCostAndTransitions(currPolicy)
            cost = g + self.discount_ * (P @ V)
            for stateInd in range(self.numStates_):
                if cost[stateInd] < minCosts[stateInd]:
                    minCosts[stateInd] = cost[stateInd]
                    policy[stateInd] = actionInd

        return policy

    
    def constructCostAndTransitions(self, policy):
        g = np.empty(self.numStates_)
        P = np.empty((self.numStates_, self.numStates_))
        for i in range(self.numStates_):
            P[i,:] = self.transitionTensor_[policy[i], i, :]
            g[i] = self.stageCosts_[i, policy[i]]
        return g, P

    def policyIteration(self, policy0, V0):
        """
        V0 is not needed theoretically but only to store the initial value function in the result st plots begin from the same point
        """
        V = V0.copy()
        policy = policy0
        policy_old = np.zeros(self.numStates_, dtype=int)
        costs = np.empty((self.numStates_, self.numActions_))
        result = [[0, V.copy().tolist(), time.time()]]
        i = 1

        print("Policy Iteration")
        while i < 1000:
            # print("Iteration: ", i)
            
            # policy evaluation
            g, P = self.constructCostAndTransitions(policy)
            V = np.linalg.solve(np.eye(self.numStates_) - self.discount_ * P, g)
            #V = g.copy() # value iteration (jacobian = identity)
            # policy improvement
            policy_old[:] = policy
            """
            # calculate costs for all actions
            for state in range(self.numStates_):
                for action in range(self.numActions_):
                    costs[state, action] = self.stageCosts_[state, action] + self.discount_ * np.dot(self.transitionTensor_[action, state, :], V)

            # extract greedy policy
            policy[:] = np.argmin(costs, axis=1)
            """
            policy[:] = self.extractGreedyPolicy(V)

            # store iteration
            result.append([i, V.copy().tolist(), time.time()])
            
            # print(f"{V} | {policy}")

            if np.array_equal(policy, policy_old):
                pass
                print("Converged after ", i, " iterations")
                return result, policy

            i += 1

    def optimisticPolicyIteration(self, policy0, V0, innerIterations):
        """
        V0 is not needed theoretically but only to store the initial value function in the result st plots begin from the same point
        """
        V = V0.copy()
        V_old = np.empty(self.numStates_)
        policy = policy0
        costs = np.empty((self.numStates_, self.numActions_))
        result = [[0, V0.copy().tolist(), time.time()]]
        i = 1

        print("Optimistic Policy Iteration")
        while True:
            # print("Iteration: ", i)
            
            # policy evaluation (V is approximated by value iteration)
            g, P = self.constructCostAndTransitions(policy)
            V_old[:] = V
            for j in range(innerIterations):
                V = g + self.discount_ * np.matmul(P, V)

            # policy improvement
            # calculate costs for all actions
            for stateInd in range(self.numStates_):
                for actionInd in range(self.numActions_):
                    costs[stateInd, actionInd] = self.stageCosts_[stateInd, actionInd] + self.discount_ * np.dot(self.transitionTensor_[actionInd, stateInd, :], V)
            
            # extract greedy policy
            policy[:] = np.argmin(costs, axis=1)

            # store iteration
            result.append([i, V.copy().tolist(), time.time()])
            
            # print(f"{V} | {policy}")

            if np.linalg.norm(V - V_old, ord=np.inf) < self.tol_:
                print("Converged after ", i, " iterations")
                return result, policy

            i += 1

    def inexactPolicyIteration(self, V0, maxIter):
        # for now exact, but algorithm according to algorithm 1 in iGMRES-PI paper
        V = V0.copy()
        V_old = np.zeros(self.numStates_)

        for k in range(maxIter):
            policy = self.extractGreedyPolicy2(V)
            g, P = self.constructCostAndTransitions(policy)
            jacobian = np.eye(self.numStates_) - self.discount_ * P
            V_old[:] = V
            V = np.linalg.solve(jacobian, g)
            if np.linalg.norm(V - V_old, ord=np.inf) < self.tol_:
                print("Converged after ", k, " iterations")
                return V, policy
