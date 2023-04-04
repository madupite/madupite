from matplotlib import pyplot as plt
import numpy as np

def plotResult(mdp, res_VI, res_PI, res_OPI):

    print("Generating plots")
    # matplotlib setup
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif"
    })
    # 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle(f'Convergence of VI/PI/OPI for $n = {mdp.numStates_}, m = {mdp.numActions_}, \\gamma = {mdp.discount_}$', fontsize="x-large")
    # plot 1: error vs iterations
    # plot 2: error vs time


    # calculate error of methods
    result_PI = [[],[]]
    result_VI = [[],[]]
    result_OPI = [[],[]]

    # optimal value is last entry of PI
    V_opt = res_PI[-1][1]
    # start time
    t0_PI = res_PI[0][2]
    t0_VI = res_VI[0][2]
    #t0_OPI = res_OPI[0][2]

    for i in range(len(res_PI)):
        v = np.array(res_PI[i][1])
        result_PI[1].append(np.linalg.norm(V_opt - v, ord=np.inf))
        result_PI[0].append(res_PI[i][2] - t0_PI)

    for i in range(len(res_VI)):
        v = np.array(res_VI[i][1])
        result_VI[1].append(np.linalg.norm(V_opt - v, ord=np.inf))
        result_VI[0].append(res_VI[i][2] - t0_VI)

    ax1.semilogy(result_VI[1], label='VI', color='blue')
    ax1.semilogy(result_PI[1], label='PI', color='red')
    ax2.plot(result_VI[0], result_VI[1], label='VI', color='blue')
    ax2.plot(result_PI[0], result_PI[1], label='PI', color='red')
    
    # plot all OPI runs (different no. of inner iterations)
    for i in range(len(res_OPI)):
        k = res_OPI[i][1] # inner iterations
        t0_OPI = res_OPI[i][0][0][2]

        for j in range(len(res_OPI[i][0])):
            v = np.array(res_OPI[i][0][j][1])
            result_OPI[1].append(np.linalg.norm(V_opt - v, ord=np.inf))
            result_OPI[0].append(res_OPI[i][0][j][2] - t0_OPI)

        ax1.semilogy(result_OPI[1], label=f"OPI $(k = {k})$")
        ax2.plot(result_OPI[0], result_OPI[1], label=f"OPI $(k = {k})$")

        result_OPI = [[],[]]
    

    # calculate convergence rate of VI $e_{i+1}/e_i$
    VI_rate = []
    VI_order = []
    for i in range(1, len(result_VI[1])):
        VI_rate.append(result_VI[1][i] / result_VI[1][i-1])
        if i > 1:
            VI_order.append(np.log(result_VI[1][i] / result_VI[1][i - 1]) / np.log(result_VI[1][i - 1] / result_VI[1][i - 2]))
    # median rate and order
    med_rate = np.median(np.array(VI_rate))
    med_order = np.median(np.array(VI_order))

    # plot settings
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('$|| V_i - V^* ||_\\infty$')
    ax1.set_ylim(1e-14, 10)
    ax1.legend()
    ax2.set_yscale('log')
    ax2.set_xlabel('Time [s]')
    ax2.set_ylabel('$|| V_i - V^* ||_\\infty$')
    ax2.set_ylim(1e-14, 10)
    ax2.legend()

    plt.figtext(0.5, 0.01, f'VI: median cvg. rate: {med_rate:.2f}, median cvg. order: {med_order:.2f}', ha="center", fontsize="small")

    plt.savefig(f'plot_{mdp.numStates_}_{mdp.numActions_}_{mdp.discount_}.pdf')
    # plt.show()

    # plot 2 convergence rate of VI
    # ax2.plot(res_VI_rate, label='rate (linear) $e_{i+1}/e_i$', color='blue')
    # ax2.plot(res_VI_order, label='order $\\log(e_{i+1}/e_i)/\\log(e_i/e_{i-1})$', color='red')
    # ax2.set_xlabel('Iteration')
    # ax2.set_ylim(0.4, 1.1)
    # ax2.set_title('Convergence rate \& order of VI')
    # ax2.legend()