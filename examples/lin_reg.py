import argparse
import numpy as np
import statsmodels.api as sm
from sklearn.datasets import fetch_california_housing
from fedstats import MetaAnalysisAggregation, LinearRegression
from fedstats.util import plot_forest


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
    reg = LinearRegression(X, y)
    reg.fit()
    return reg.get_result()


def main(save_plot=False):
    Xs, ys = load_split_data()
    local_datasets = list(zip(Xs, ys))

    # fit models and get results
    results_nodes = list(map(lambda tup: fit_local_model(*tup), local_datasets))

    # aggregate results
    agg = MetaAnalysisAggregation(results_nodes)
    agg.aggregate_results()
    results_agg = agg.get_aggregated_results()

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
    plot_forest(
        data_trim,
        ylabels=housing.feature_names,
        names=["aggregated", "reference"],
        save_plot=save_plot,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Save the Plot?")
    parser.add_argument(
        "--saveplot", type=bool, default=False, help="If true, plot will be saved."
    )
    main(save_plot=parser.parse_args().saveplot)
