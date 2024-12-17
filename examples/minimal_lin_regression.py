
"""
Simple example on how to aggregate one single statistical estimator using a simple linear regression
"""
import argparse
import numpy as np
from fedstats.aggregation import MetaAnalysisAggregator 
import scipy.stats as stats


def make_local_data(n):
    x = np.random.normal(size=n)
    y = x + np.random.normal(size=n)
    return x, y


def run_regression(x, y):
    mod = stats.linregress(x,y)
    return mod.slope.item(), mod.stderr.item()  # type: ignore


def run_local_model(n):
    x,y = make_local_data(n)
    return run_regression(x,y)


def main(num_clients, num_obs_each):
    res = [run_local_model(num_obs_each) for _ in range(num_clients)]
    meta_analysis = MetaAnalysisAggregator(res)
    meta_analysis.aggregate_results()
    results = meta_analysis.get_results()
    print(f"Aggregated Results calculated on {num_clients} clients:")
    print(f"Aggregated effect: {results['aggregated_results']:.4f}")
    print(f"95% Confidence Interval: ({results['confidence_interval'][0]:.4f}, {results['confidence_interval'][1]:.4f})")
    # print(f"Q-Statistic (Heterogeneity): {results['q_statistic']:.4f}")




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Number of clients and observations.")
    parser.add_argument("--clients", type=int, default=5, help="Number of clients.")
    parser.add_argument("--obs", type=int, default=100, help="Number of observations at each client.")
    args = parser.parse_args()
    main(args.clients, args.obs)



