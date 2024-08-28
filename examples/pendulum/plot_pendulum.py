import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa


# Function definitions
def reshape_data(data, shape):
    return np.rot90(data.reshape(shape), k=1)


def create_tick_labels(values, num_ticks):
    indices = np.linspace(0, len(values) - 1, num_ticks, dtype=int)
    return indices, [f"{val:.2f}" for val in values[indices]]


def plot_heatmap(ax, data, title, x_val, xd_val):
    im = ax.imshow(data, cmap="jet", interpolation="nearest")
    ax.set_title(title)
    ax.set_ylabel(r"$\dot{x}$")
    ax.set_xlabel("$x$")
    yticks, yticklabels = create_tick_labels(xd_val[::-1], 5)
    xticks, xticklabels = create_tick_labels(x_val, 5)
    ax.set_yticks(yticks)
    ax.set_xticks(xticks)
    ax.set_yticklabels(yticklabels)
    ax.set_xticklabels(xticklabels)
    return im


def main():
    # Load data from the output files
    data = np.loadtxt("data/pend_cost.out")
    n = np.sqrt(len(data)).astype(int)
    costs = reshape_data(data, (n, n))
    policy = np.loadtxt("data/pend_policy.out", dtype=int)

    x_vals = np.linspace(0, 2 * np.pi, n)  # Recreate the x values
    xd_vals = np.linspace(-10.0, 10.0, n)  # Recreate the xd values
    a_vals = np.linspace(-3.0, 3.0, 9)  # Recreate the action values
    user_input = reshape_data(a_vals[policy], (n, n))

    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    im1 = plot_heatmap(ax1, costs, "Costs", x_vals, xd_vals)
    im2 = plot_heatmap(ax2, user_input, "Policy", x_vals, xd_vals)
    fig.colorbar(im1, ax=ax1)
    fig.colorbar(im2, ax=ax2)
    plt.savefig("pendulum.png", dpi=300)


if __name__ == "__main__":
    main()
