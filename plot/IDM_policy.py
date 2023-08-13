import matplotlib.pyplot as plt
import numpy as np


# plot policy of infectious disease model (action ind on y-axis), state ind on x-axis
# policy: ../cmake-build-debug/IDM_out/policy.out


def plot_policy(policy, title):
    plt.plot(policy)
    plt.title(title)
    plt.xlabel('Number of Suspectibles')
    plt.ylabel('Action index')
    plt.yticks(np.arange(0, 21, 1))
    plt.xticks(np.arange(0, 2001, 250))
    plt.savefig('IDM_policy.png', dpi=300)

if __name__ == '__main__':
    policy = np.loadtxt('../cmake-build-debug/IDM_out/policy.out')
    plot_policy(policy, 'Policy of Infectious Disease Model')