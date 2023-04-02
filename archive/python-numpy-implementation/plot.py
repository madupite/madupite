from matplotlib import pyplot as plt
import numpy as np

def plotResult(mdp, res_VI, res_PI, res_OPI):


    # calculate error of methods
    result_PI = [[],[]]
    result_VI = [[],[]]
    result_OPI = [[],[]]

    # optimal value is last entry of PI
    V_opt = res_PI[-1][1]
    # start time
    t0_PI = res_PI[0][2]
    t0_VI = res_VI[0][2]
    t0_OPI = res_OPI[0][2]

    for i in range(len(res_PI)):
        v = np.array(res_PI[i][1])
        result_PI[1].append(np.linalg.norm(V_opt - v, ord=np.inf))
        result_PI[0].append(res_PI[i][2] - t0_PI)

    for i in range(len(res_VI)):
        v = np.array(res_VI[i][1])
        result_VI[1].append(np.linalg.norm(V_opt - v, ord=np.inf))
        result_VI[0].append(res_VI[i][2] - t0_VI)

    for i in range(len(res_OPI)):
        v = np.array(res_OPI[i][1])
        result_OPI[1].append(np.linalg.norm(V_opt - v, ord=np.inf))
        result_OPI[0].append(res_OPI[i][2] - t0_OPI)
    
    # # calculate convergence *order* of VI $log(e_{i+1}/e_i)/log(e_i/e_{i-1})$
    # res_VI_order = np.empty(len(result_VI) - 2)
    # for i in range(2, len(result_VI)):
    #     res_VI_order[i - 2] = np.log(result_VI[i] / result_VI[i - 1]) / np.log(result_VI[i - 1] / result_VI[i - 2])

    # # calculate convergence rate according to NumCSE for linear cvg (Def. 8.2.2.1) $\sup e_{i+1} / e_i$
    # res_VI_rate = np.empty(len(result_VI) - 1)
    # for i in range(1, len(result_VI)):
    #     res_VI_rate[i - 1] = result_VI[i] / result_VI[i - 1]


    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.family": "palatino",
    })

    # 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle(f'VI/PI/OPI for $n = {mdp.numStates_}, m = {mdp.numActions_}, \\gamma = {mdp.discount_}$', fontsize="x-large")

    # plot 1 VI vs PI vs OPI (per iteration)
    ax1.semilogy(result_VI[1], label='VI', color='blue')
    ax1.semilogy(result_PI[1], label='PI', color='red')
    ax1.semilogy(result_OPI[1], label='OPI', color='green')
    #ax1.set_yscale('symlog')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('$|| V_i - V^* ||_\\infty$')
    ax1.set_title('Convergence of VI, PI and OPI')
    ax1.set_ylim(1e-14, 10)
    ax1.legend()

    # plot 2 convergence rate of VI
    # ax2.plot(res_VI_rate, label='rate (linear) $e_{i+1}/e_i$', color='blue')
    # ax2.plot(res_VI_order, label='order $\\log(e_{i+1}/e_i)/\\log(e_i/e_{i-1})$', color='red')
    # ax2.set_xlabel('Iteration')
    # ax2.set_ylim(0.4, 1.1)
    # ax2.set_title('Convergence rate \& order of VI')
    # ax2.legend()

    # plot 2 convergence (log) vs time
    ax2.plot(result_VI[0], result_VI[1], label='VI', color='blue')
    ax2.plot(result_PI[0], result_PI[1], label='PI', color='red')
    ax2.plot(result_OPI[0], result_OPI[1], label='OPI', color='green')
    ax2.set_yscale('log')
    ax2.set_xlabel('Time [s]')
    ax2.set_ylabel('$|| V_i - V^* ||_\\infty$')
    ax2.set_title('Convergence of VI, PI and OPI')
    ax2.set_ylim(1e-14, 10)
    ax2.legend()


    plt.savefig('plot.pdf')
    # plt.show()
