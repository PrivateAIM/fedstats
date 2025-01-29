import numpy as np
import matplotlib.pyplot as plt


def plot_forest(
    data,
    ylabels=None,
    colors=None,
    names=None,
    alpha=None,
    save_plot=False,
    filename="Forestplot",
):
    for i, (point_estimates, lower_bounds, upper_bounds) in enumerate(data):
        if len(point_estimates) != len(lower_bounds) or len(point_estimates) != len(
            upper_bounds
        ):
            raise ValueError(
                f"Arrays for point estimates, lower bounds, and upper bounds must have the same length for dataset {i}."
            )

    n_points = len(data[0][0])

    if ylabels is None:
        ylabels = [f"Point {i + 1}" for i in range(n_points)]

    if len(ylabels) != n_points:
        raise ValueError(
            "y - Labels list must have the same length as the number of points."
        )

    # Generate colors for each dataset
    if colors is None:
        colors = plt.cm.cividis(np.linspace(0, 1, len(data)))

    if alpha is None:
        alpha = [1 for _ in range(len(data))]

    if names is None:
        names = [str(i + 1) for i in range(len(data))]

    # Create the plot
    plt.figure(figsize=(8, 0.5 * n_points))

    y_positions = np.arange(n_points)
    jitter_offsets = np.linspace(
        0.15, -0.15, len(data)
    )  # Create small offsets for jittering

    for idx, (point_estimates, lower_bounds, upper_bounds) in enumerate(data):
        jittered_positions = (
            y_positions + jitter_offsets[idx]
        )  # Apply jitter to y-positions
        plt.errorbar(
            point_estimates,
            jittered_positions,
            xerr=[point_estimates - lower_bounds, upper_bounds - point_estimates],
            fmt="o",
            color=colors[idx],
            ecolor=colors[idx],
            capsize=4,
            label=names[idx],
            alpha=alpha[idx],
            elinewidth=3,
            markeredgewidth=3,
        )

    plt.yticks(y_positions, ylabels)

    # Add grid, labels, and a vertical line at 0 for reference
    plt.axvline(x=0, color="red", linestyle="--", linewidth=0.8)
    plt.grid(axis="y", linestyle="--", linewidth=0.5)
    plt.xlabel("Estimate")
    plt.title("Comparison of results")
    plt.legend()

    plt.tight_layout()
    if save_plot:
        filename = f"{filename}_{datetime.now().strftime('%d-%m-%Y')}.png"
        plt.savefig(filename)
    else:
        plt.show()
