import pandas as pd
from flame.star import StarAggregator, StarAnalyzer, StarModelTester
from lifelines import CoxPHFitter
from lifelines.datasets import load_rossi

from fedstats import MetaAnalysisAggregation


class LocalCoxModel(StarAnalyzer):
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
        cph = CoxPHFitter()
        cph.fit(data, duration_col="week", event_col="arrest")
        est, sds = cph.params_.to_list(), (cph.standard_errors_**2).to_list()
        return list(zip(est, sds, strict=False))


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
        # fit the model on the full data set for comparison
        data = load_rossi()
        cph = CoxPHFitter()
        cph.fit(data, duration_col="week", event_col="arrest")
        res_full_data = pd.DataFrame(
            {
                "type": "full_data",
                "name": cph.params_.index,
                "coef": cph.params_.to_numpy(),
                "ci_lower": cph.confidence_intervals_.iloc[:, 0].to_numpy(),
                "ci_upper": cph.confidence_intervals_.iloc[:, 1].to_numpy(),
            }
        )

        # aggregate results
        aggregator = MetaAnalysisAggregation(analysis_results)
        aggregator.aggregate_results()
        results_aggregated = aggregator.get_aggregated_results()

        res_aggregated = pd.DataFrame(
            {
                "type": "aggregated",
                "name": cph.params_.index,
                "coef": results_aggregated["aggregated_results"],
                "ci_lower": results_aggregated["confidence_interval"][:, 0], # type: ignore
                "ci_upper": results_aggregated["confidence_interval"][:, 1], # type: ignore
            }
        )

        return pd.concat((res_full_data, res_aggregated))

    def has_converged(self, result, last_result):
        """
        Determines if the aggregation process has converged.

        :param result: The current aggregated result.
        :param last_result: The aggregated result from the previous iteration.
        :param num_iterations: The number of iterations completed so far.
        :return: True if the aggregation has converged; False to continue iterations.
        """
        return True  # Return True to indicate convergence in this simple analysis


def main():
    """
    Sets up and initiates the distributed analysis using the FLAME components.

    - Defines the custom analyzer and aggregator classes.
    - Specifies the type of data and queries to execute.
    - Configures analysis parameters like iteration behavior and output format.
    """

    data = load_rossi()
    # shuffle and split the data into 2 parts to simulate 2 different data sources
    data = data.sample(frac=1, random_state=42).reset_index(drop=True)
    split_data = [data.iloc[: len(data) // 2], data.iloc[len(data) // 2 :]]


    StarModelTester(
        data_splits=split_data,  # List of data splits for each analyzer node (not needed for this example as dummy data is generated within the analyzer)
        analyzer=LocalCoxModel,  # Custom analyzer class (must inherit from StarAnalyzer)
        aggregator=ResultsAggregator,  # Custom aggregator class (must inherit from StarAggregator)
        data_type="s3",  # Type of data source ('fhir' or 's3')
        simple_analysis=True,  # True for single-iteration; False for multi-iterative analysis
        output_type="str",  # Output format for the final result ('str', 'bytes', or 'pickle')
        analyzer_kwargs=None,  # Additional keyword arguments for the custom analyzer constructor (i.e. MyAnalyzer)
        aggregator_kwargs=None,  # Additional keyword arguments for the custom aggregator constructor (i.e. MyAggregator)
    )


if __name__ == "__main__":
    main()
