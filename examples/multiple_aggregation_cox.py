"""
Example to run Cox proportional Hazard model and aggregate estimators using MetaAnalysisAggregatorCollection.
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
from lifelines import CoxPHFitter
from lifelines.datasets import load_rossi

from fedstats import MetaAnalysisAggregation
from fedstats.util import plot_forest


def load_split_data(num_clients=2, random_state=42):
    data = load_rossi()

    shuffled_data = data.sample(frac=1, random_state=random_state).reset_index(drop=True)

    chunk_size = len(shuffled_data) // num_clients
    remainder = len(shuffled_data) % num_clients

    sub_datasets = []
    start_idx = 0
    for i in range(num_clients):
        end_idx = start_idx + chunk_size + (1 if i < remainder else 0)
        sub_datasets.append(shuffled_data.iloc[start_idx:end_idx])
        start_idx = end_idx

    return sub_datasets


def apply_cox(data_chunk):
    cph = CoxPHFitter()
    cph.fit(data_chunk, duration_col="week", event_col="arrest")
    est, sds = cph.params_.to_list(), (cph.standard_errors_**2).to_list()
    return list(zip(est, sds, strict=False))


def make_ests(x):
    x = np.array(x)
    m = x[:, 0]
    return m, m - 1.96 * x[:, 1], m + 1.96 * x[:, 1]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Save the Plot?")
    parser.add_argument("--saveplot", type=bool, default=False, help="If true, plot will be saved.")

    datasets = load_split_data()
    results = list(map(apply_cox, datasets))

    # Aggregation via meta analysis approach
    aggregator = MetaAnalysisAggregation(results)
    aggregator.aggregate_results()
    results_agg = aggregator.get_aggregated_results()

    # single result for effect and ci on each node
    results_nodes = list(map(make_ests, results))

    # compare with results on full dataset
    cph = CoxPHFitter()
    cph.fit(load_rossi(), duration_col="week", event_col="arrest")
    eff_glob = cph.params_.to_numpy()
    cil, ciu = (
        cph.confidence_intervals_.iloc[:, 0].to_numpy(),
        cph.confidence_intervals_.iloc[:, 1].to_numpy(),
    )

    data = [
        (
            results_agg["aggregated_results"],
            results_agg["confidence_interval"][:, 0],
            results_agg["confidence_interval"][:, 1],
        ),
        (eff_glob, cil, ciu),
        results_nodes[0],
        results_nodes[1],
    ]

    # Plot
    plt.style.use("seaborn-v0_8")
    plot_forest(
        data,
        ylabels=cph.params_.index.to_list(),
        colors=["darkblue", "darkorange", "darkgray", "darkgray"],
        names=["aggregated", "pooled data", "local data1", "local data2"],
        alpha=[0.6, 0.6, 1, 1],
        save_plot=parser.parse_args().saveplot,
    )
