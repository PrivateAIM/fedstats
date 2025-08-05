from flame.star import StarModel, StarAnalyzer, StarAggregator
from fedstats import MetaAnalysisAggregation, LinearRegression
import numpy as np


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

        ## load data
        # required to take first element as we have one data store and the list has therefore 1 entry
        data = data[0]

        feature_name = [name for name in data if name.startswith("X_node")][0]
        target_name = [name for name in data if name.startswith("y_node")][0]

        self.targets = np.fromstring(data[target_name], sep=" ", dtype=float)
        self.features = np.fromstring(data[feature_name], sep=" ", dtype=float).reshape(
            self.targets.size, -1
        )

        # run regression
        reg = LinearRegression(self.features, self.targets)
        reg.fit()
        return reg.get_result()


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
        return agg.get_results()

    def has_converged(self, result, last_result, num_iterations):
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
    StarModel(
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
    main()
