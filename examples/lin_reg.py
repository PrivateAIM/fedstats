import numpy as np
import statsmodels.api as sm
from sklearn.datasets import fetch_california_housing
import matplotlib.pyplot as plt
from fedstats.models.LocalLinearRegression import LocalLinearRegression
from fedstats.aggregation.meta_analysis import MetaAnalysisAggregator


def load_split_data(num_clients=5, random_state=42):
    housing = fetch_california_housing()
    X = housing.data
    y = housing.target
    n, p = X.shape
    rng = np.random.default_rng(random_state)
    index = np.arange(n)
    rng.shuffle(index)
    X = X[index, :]
    y = y[index]
    X = np.array_split(X, num_clients)
    y = np.array_split(y, num_clients)
    return X, y


def fit_local_model(X, y):
    reg = LocalLinearRegression(X, y)
    reg.fit()
    return reg.get_result()


def plot_forest(
    data, ylabels=None, colors=None, names=None, alpha=None, save_plot=False
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
        filename = f"results_CoxPH_{datetime.now().strftime('%d-%m-%Y')}.png"
        plt.savefig(filename)
    else:
        plt.show()


def main(save_plot=False):
    Xs, ys = load_split_data()
    local_datasets = list(zip(Xs, ys))

    # fit models and get results
    results_nodes = list(map(lambda tup: fit_local_model(*tup), local_datasets))

    # aggregate results
    agg = MetaAnalysisAggregator(results_nodes)
    agg.aggregate_results()
    results_agg = agg.get_results()

    # compare with regression on global data
    housing = fetch_california_housing()
    X, y = housing.data, housing.target
    X_ = np.c_[np.ones(y.shape), X]
    model = sm.OLS(y, X_)
    result = model.fit()
    coefs = result.params
    sds = result.bse
    cil = coefs - 1.96 * sds
    ciu = coefs + 1.96 * sds

    data = [
        (
            results_agg["aggregated_results"],
            results_agg["confidence_interval"][:, 0],
            results_agg["confidence_interval"][:, 1],
        ),
        (coefs, cil, ciu),
    ]

    print("=== Results aggregatred ===")
    print(results_agg)

    print("=== Results on full dataset ===")
    print("Effects:")
    print(coefs)

    print("CI:")
    print(list(zip(cil, ciu)))

    # get rid of intercept to make a nice plot
    data_trim = list(map(lambda x: list(map(lambda y: y[1:], x)), data))
    plot_forest(data_trim, ylabels=housing.feature_names, names=["aggregated","reference"] save_plot=save_plot)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Save the Plot?")
    parser.add_argument(
        "--saveplot", type=bool, default=False, help="If true, plot will be saved."
    )
    main(save_plot=parser.parse_args().saveplot)
    main()
