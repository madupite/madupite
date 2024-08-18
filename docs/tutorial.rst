Tutorial
===============

Let's start with a minimal example that demonstrates how to quickly load and solve for the optimal value function and policy of a Markov Decision Process (MDP) using ``madupite``.

We define an MDP as a tuple :math:`(\mathcal{S}, \mathcal{A}, P, g, \gamma)`, where: 
- :math:`\mathcal{S} = \{0, 1, \dots, n-1\}` is the set of states,
- :math:`\mathcal{A} = \{0, 1, \dots, m-1\}` is the set of actions,
- :math:`P : \mathcal{S} \times \mathcal{A} \times \mathcal{S} \to [0, 1]` is the transition probability function, where :math:`P(s, a, s') = \text{Pr}(s_{t+1} = s' | s_t = s, a_t = a)` and :math:`\sum_{s' \in \mathcal{S}} P(s, a, s') = 1` for all :math:`s \in \mathcal{S}` and :math:`a \in \mathcal{A}`,
- :math:`g : \mathcal{S} \times \mathcal{A} \to \mathbb{R}` is the stage cost function (or reward in an alternative but equivalent formulation),
- :math:`\gamma \in (0, 1)` is the discount factor.


Example 1
----------

We start with a simple example: We have an agent that lives on a periodic 1-dimensional line of length 50. At each time step, the agent has to choose between moving to the left, to the right or staying in place. The goal is to move to the state with index 42. We want to find the optimal policy that minimizes the expected number of steps to reach the goal.

.. image:: _static/tutorial_ex1.pdf
    :align: center

Note that we use 0-based indexing to facilitate translating the mathematical model into code later.

To do so, we first define the state space :math:`\mathcal{S} = \{0, 1, \dots, 49\}` and the action space :math:`\mathcal{A} = \{0, 1, 2\}` where 0 means staying in place, 1 means moving to the left and 2 means moving to the right. 

For the usage in ``madupite``, we need to provide a transition probability function that returns for a given state-action pair :math:`(s, a)` a list of probabilities and a list of corresponding next states. In this case, the model is deterministic and thus the probabilities are always 1. The transition probability function is defined as follows:

.. math::

    P(s, a) = \begin{cases}
        ((1), (s)) & \text{if } a = 0 \\
        ((1), ((s-1) \mod 50)) & \text{if } a = 1 \\
        ((1), ((s+1) \mod 50)) & \text{if } a = 2
    \end{cases}

Stochasticity can be introduced by making the movements non-deterministic. Let's say the when the agent moves to left or right, there is a 10% chance that they will stay in place instead. And when the agent stays in place, there is a 10% chance each that they will move to the left or right. The transition probability function is defined as follows:

.. math::

    P(s, a) = \begin{cases}
        ((0.1, 0.8, 0.1), ((s-1) \mod 50, s, (s+1) \mod 50)) & \text{if } a = 0 \\
        ((0.1, 0.9), (s, (s-1) \mod 50)) & \text{if } a = 1 \\
        ((0.1, 0.9), (s, (s+1) \mod 50)) & \text{if } a = 2
    \end{cases}

Instead of defining a stage cost function, we define a reward function and set the optimization mode to maximize instead of minimize later. The reward function is defined as follows:

.. math::
    r(s, a) = \begin{cases}
        1 & \text{if } s = 42 \\
        0 & \text{otherwise}
    \end{cases}


