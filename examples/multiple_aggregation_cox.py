"""
Example to run Cox proportional Hazard model and aggregate estimators using MetaAnalysisAggregatorCollection.
"""
import argparse
from datetime import datetime
import numpy as np
from lifelines import CoxPHFitter
from lifelines.datasets import load_rossi
import matplotlib.pyplot as plt
from fedstats.aggregation.meta_analysis import MetaAnalysisAggregatorCollection

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
    cph.fit(data_chunk, duration_col='week', event_col='arrest')
    est, sds = cph.params_.to_list(), (cph.standard_errors_**2).to_list()
    return list(zip(est, sds))


def make_ests(x):
    x = np.array(x)
    m = x[:,0]
    return m, m-1.96*x[:,1], m+1.96*x[:,1]


def plot_forest(data, ylabels=None, colors=None, names=None, alpha=None, save_plot=False):
    for i, (point_estimates, lower_bounds, upper_bounds) in enumerate(data):
        if len(point_estimates) != len(lower_bounds) or len(point_estimates) != len(upper_bounds):
            raise ValueError(f"Arrays for point estimates, lower bounds, and upper bounds must have the same length for dataset {i}.")

    n_points = len(data[0][0])

    if ylabels is None:
        ylabels = [f"Point {i+1}" for i in range(n_points)]

    if len(ylabels) != n_points:
        raise ValueError("y - Labels list must have the same length as the number of points.")

    # Generate colors for each dataset
    if colors is None:
        colors = plt.cm.cividis(np.linspace(0, 1, len(data)))

    if alpha is None:
        alpha = [1 for _ in range(len(data))]

    if names is None:
        names = [str(i+1) for i in range(len(data))]

    # Create the plot
    plt.figure(figsize=(8, 0.5 * n_points))
    
    y_positions = np.arange(n_points)
    jitter_offsets = np.linspace(0.15, -0.15, len(data))  # Create small offsets for jittering

    for idx, (point_estimates, lower_bounds, upper_bounds) in enumerate(data):
        jittered_positions = y_positions + jitter_offsets[idx]  # Apply jitter to y-positions
        plt.errorbar(point_estimates, jittered_positions, 
                     xerr=[point_estimates - lower_bounds, upper_bounds - point_estimates], 
                     fmt='o', color=colors[idx], ecolor=colors[idx], capsize=4,
                     label=names[idx],
                     alpha=alpha[idx],
                     elinewidth=3,
                     markeredgewidth=3)
    
    plt.yticks(y_positions, ylabels)

    # Add grid, labels, and a vertical line at 0 for reference
    plt.axvline(x=0, color='red', linestyle='--', linewidth=0.8)
    plt.grid(axis='y', linestyle='--', linewidth=0.5)
    plt.xlabel('Estimate')
    plt.title('Comparison of results')
    plt.legend()
    
    plt.tight_layout()
    if save_plot:
        filename = f"results_CoxPH_{datetime.now().strftime("%d-%m-%Y")}.png"
        plt.savefig(filename)
    else:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Save the Plot?")
    parser.add_argument("--saveplot", type=bool, default=False, help="If true, plot will be saved.")


    datasets = load_split_data()
    results = list(map(apply_cox, datasets))

    # Aggregation via meta analysis approach 
    aggregator = MetaAnalysisAggregatorCollection(results)
    aggregator.aggregate_results()
    results_agg = aggregator.get_results()

    # single result for effect and ci on each node
    results_nodes = list(map(make_ests, results))


    # compare with results on full dataset
    cph = CoxPHFitter()
    cph.fit(load_rossi(), duration_col='week', event_col='arrest')
    eff_glob = cph.params_.to_numpy()
    cil, ciu = cph.confidence_intervals_.iloc[:,0].to_numpy(), cph.confidence_intervals_.iloc[:,1].to_numpy()


    data = [
        (results_agg["aggregated_results"], results_agg["confidence_interval"][:,0], results_agg["confidence_interval"][:,1]),
        (eff_glob, cil, ciu),
        results_nodes[0],
        results_nodes[1]
    ]
    
    # Plot
    plt.style.use('seaborn-v0_8')
    plot_forest(data, 
                ylabels=cph.params_.index.to_list(),
                colors=["darkblue", "darkorange", "darkgray", "darkgray"],
                names=["aggregated", "pooled data", "local data1", "local data2"],
                alpha=[0.6,0.6,1,1],
                save_plot=parser.parse_args().saveplot)

