"""
Simple example on how to aggregate one single statistical estimator using a simple linear regression
"""

import argparse

import numpy as np
import scipy.stats as stats

from fedstats import AverageAggregation, MetaAnalysisAggregation


def make_local_data(n, seed):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    y = x + rng.normal(size=n)
    return x, y


def run_regression(x, y):
    mod = stats.linregress(x, y)
    return mod.slope.item(), mod.stderr.item()  # type: ignore


def run_local_model(n, seed):
    x, y = make_local_data(n, seed)
    return run_regression(x, y)


def main(num_clients, num_obs_each, seed):
    rng = np.random.default_rng(seed)
    seeds = rng.integers(0, 99999, size=num_clients)
    res = [run_local_model(num_obs_each, seed) for seed in seeds]

    # aggregation via meta analysis
    meta_analysis = MetaAnalysisAggregation(res)
    meta_analysis.aggregate_results()
    results_ma = meta_analysis.get_aggregated_results()

    # aggregation via average
    # modify results: overwrite standard errors with sample size
    res2 = [(res_k[0], num_obs_each) for res_k in res]
    average_agg = AverageAggregation(res2)
    average_agg.aggregate_results()
    results_avg = average_agg.get_aggregated_results()

    print("=============Results=============")
    print("Results using meta analysis")
    print(f"Aggregated Results calculated on {num_clients} clients:")
    print(f"Aggregated effect: {results_ma['aggregated_results']:.4f}")
    ci = results_ma["confidence_interval"]
    print(f"95% Confidence Interval: ({ci[0]:.4f}, {ci[1]:.4f})")  # type: ignore
    # print(f"Q-Statistic (Heterogeneity): {results['q_statistic']:.4f}")

    print("\n")

    print("Results using average")
    print(f"Aggregated Results calculated on {num_clients} clients:")
    print(f"Aggregated effect: {results_avg['aggregated_results']:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Number of clients and observations.")
    parser.add_argument("--clients", type=int, default=5, help="Number of clients.")
    parser.add_argument("--obs", type=int, default=100, help="Number of observations at each client.")
    parser.add_argument("--seed", type=int, default=42, help="Seed to generate data.")
    args = parser.parse_args()
    main(args.clients, args.obs, args.seed)
