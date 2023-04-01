from matplotlib import pyplot as plt
import numpy as np

def main():
    # import data from file
    res_PI = np.fromfile('cmake-build-release/res_PI.txt')
    res_VI = np.loadtxt('cmake-build-release/res_VI.txt')
    res_OPI = np.loadtxt('cmake-build-release/res_OPI.txt')

    # plot use tex
    plt.rcParams.update({
        "text.usetex": True
    })
    plt.semilogy(res_PI, label='PI', color='red')
    plt.semilogy(res_VI, label='VI', color='blue')
    plt.semilogy(res_OPI, label='OPI', color='green')
    plt.xlabel('Iteration')
    plt.ylabel('$\\vert V_i - V^* \\vert_\\infty$')
    plt.ylim(1e-6, 10)
    plt.legend()
    plt.savefig('cmake-build-release/plot.pdf')






if __name__ == '__main__':
    main()
