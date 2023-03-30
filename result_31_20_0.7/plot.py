from matplotlib import pyplot as plt
import numpy as np

def main():
    # import data from file
    res_PI = np.loadtxt('res_PI.txt')
    res_VI = np.loadtxt('res_VI.txt')
    res_OPI = np.loadtxt('res_OPI.txt')

    # calculate convergence *order* of VI $log(e_{i+1}/e_i)/log(e_i/e_{i-1})$
    res_VI_order = np.zeros(res_VI.shape[0] - 2)
    for i in range(2, res_VI.shape[0]):
        res_VI_order[i - 2] = np.log(res_VI[i] / res_VI[i - 1]) / np.log(res_VI[i - 1] / res_VI[i - 2])

    # calculate convergence rate according to NumCSE for linear cvg (Def. 8.2.2.1) $\sup e_{i+1} / e_i$
    res_VI_rate = np.zeros(res_VI.shape[0] - 1)
    for i in range(1, res_VI.shape[0]):
        res_VI_rate[i - 1] = res_VI[i] / res_VI[i - 1]


    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif"
    })

    # 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle('VI/PI/OPI for $n = 31, m = 20, \\gamma = 0.7$', fontsize="x-large")

    # plot 1 VI vs PI vs OPI
    ax1.semilogy(res_VI, label='VI', color='blue')
    ax1.semilogy(res_PI, label='PI', color='red')
    ax1.semilogy(res_OPI, label='OPI ($K=15$)', color='green')
    #ax1.set_yscale('symlog')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('$|| V_i - V^* ||_\\infty$')
    ax1.set_title('Convergence of VI, PI and OPI')
    ax1.set_ylim(1e-14, 10)
    ax1.legend()

    # plot 2 convergence rate of VI
    ax2.plot(res_VI_rate, label='rate (linear) $e_{i+1}/e_i$', color='blue')
    ax2.plot(res_VI_order, label='order $\\log(e_{i+1}/e_i)/\\log(e_i/e_{i-1})$', color='red')
    ax2.set_xlabel('Iteration')
    ax2.set_ylim(0.4, 1.2)
    ax2.set_title('Convergence rate \& order of VI')
    ax2.legend()

    plt.savefig('plot.pdf')
    # plt.show()


if __name__ == '__main__':
    main()
