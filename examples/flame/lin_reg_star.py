# FLAME implementation of the basic linear regression example in examples/standalone/minimal_lin_regression.py

import argparse

import numpy as np
from flame.star import StarAggregator, StarAnalyzer, StarModelTester

from examples.standalone.minimal_lin_regression import make_local_data, run_regression
from fedstats import MetaAnalysisAggregation


class LocalLinearModel(StarAnalyzer):
    def __init__(self, flame):
        """
        Initializes the custom Analyzer node.

        :param flame: Instance of FlameCoreSDK to interact with the FLAME components.
        """
        super().__init__(flame)  # Connects this analyzer to the FLAME components

    def analysis_method(self, data, aggregator_results):
        """
        Performs analysis on the retrieved data from data sources.

        :param data: A list of dictionaries containing the data from each data source.
                     - Each dictionary corresponds to a data source.
                     - Keys are the queries executed, and values are the results (dict for FHIR, str for S3).
        :param aggregator_results: Results from the aggregator in previous iterations.
                                   - None in the first iteration.
                                   - Contains the result from the aggregator's aggregation_method in subsequent iterations.
        :return: Any result of your analysis on one node (ex. patient count).
        """
        return run_regression(data[0], data[1])


class ResultsAggregator(StarAggregator):
    def __init__(self, flame):
        """
        Initializes the custom Aggregator node.

        :param flame: Instance of FlameCoreSDK to interact with the FLAME components.
        """
        super().__init__(flame)  # Connects this aggregator to the FLAME components

    def aggregation_method(self, analysis_results):
        """
        Aggregates the results received from all analyzer nodes.

        :param analysis_results: A list of analysis results from each analyzer node.
        :return: The aggregated result (e.g., total patient count across all analyzers).
        """
        agg = MetaAnalysisAggregation(analysis_results)
        agg.aggregate_results()
        return agg.get_aggregated_results()

    def has_converged(self, result, last_result):
        """
        Determines if the aggregation process has converged.

        :param result: The current aggregated result.
        :param last_result: The aggregated result from the previous iteration.
        :param num_iterations: The number of iterations completed so far.
        :return: True if the aggregation has converged; False to continue iterations.
        """
        return True  # Return True to indicate convergence in this simple analysis


def main(num_clients, num_obs_each, seed):
    """
    Sets up and initiates the distributed analysis using the FLAME components.

    - Defines the custom analyzer and aggregator classes.
    - Specifies the type of data and queries to execute.
    - Configures analysis parameters like iteration behavior and output format.
    """
    rng = np.random.default_rng(seed=seed)
    seeds = rng.integers(0, 99999, size=num_clients)

    datasets = [make_local_data(num_obs_each, seed) for seed in seeds]

    StarModelTester(
        data_splits=datasets,  # List of data splits for each analyzer node (not needed for this example as dummy data is generated within the analyzer)
        analyzer=LocalLinearModel,  # Custom analyzer class (must inherit from StarAnalyzer)
        aggregator=ResultsAggregator,  # Custom aggregator class (must inherit from StarAggregator)
        data_type="s3",  # Type of data source ('fhir' or 's3')
        # query="Patient?_summary=count",  # Query or list of queries to retrieve data
        simple_analysis=True,  # True for single-iteration; False for multi-iterative analysis
        output_type="str",  # Output format for the final result ('str', 'bytes', or 'pickle')
        analyzer_kwargs=None,  # Additional keyword arguments for the custom analyzer constructor (i.e. MyAnalyzer)
        aggregator_kwargs=None,  # Additional keyword arguments for the custom aggregator constructor (i.e. MyAggregator)
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Number of clients and observations.")
    parser.add_argument("--clients", type=int, default=5, help="Number of clients.")
    parser.add_argument("--obs", type=int, default=100, help="Number of observations at each client.")
    parser.add_argument("--seed", type=int, default=42, help="Seed to generate data.")
    args = parser.parse_args()
    main(args.clients, args.obs, args.seed)
