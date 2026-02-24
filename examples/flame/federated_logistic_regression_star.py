import numpy as np
from flame.star import StarAggregator, StarAnalyzer, StarModelTester

from fedstats import FederatedGLM, PartialFisherScoring
from fedstats.util import simulate_logistic_regression


class LocalFisherScoring(StarAnalyzer):
    def __init__(self, flame):
        """Initializes local analyzer node."""
        super().__init__(flame)  # Connects this analyzer to the FLAME components

    def analysis_method(self, data, aggregator_results):
        """
        Runs local parts of the federated fisher scoring
        Fits score vectore and fisher information matrix on current values from aggregator results
        aggregator_results should be a list with one element. This element is a tuple 2 elements:
        1. Aggregation results (np.ndarray) 2. convergence flag.
        """
        print()
        print(f"------ ANALYSIS in {self.id} ------")

        self.x, self.y = data[0], data[1]

        self.local_model_parts = PartialFisherScoring(
            self.x, self.y, family="binomial", fit_intercept=False
        )

        # first iteration, aggregator gives no results and therefore None, use local inital values. In subsequent iterations, use aggregator results.
        if aggregator_results is None:
            print(f"Initial values of beta: {self.local_model_parts.beta}")

            aggregator_results = (self.local_model_parts.beta, False)


        print(f"Aggregator results (it. {self.num_iterations}) are: {aggregator_results}")
        # if condition checks, converged flag. In the case of convergence, return the result
        if not aggregator_results[1]:
            aggregator_results = aggregator_results[0]
            self.local_model_parts.set_coefs(aggregator_results)
            fisher_scoring_parts = self.local_model_parts.calc_fisher_scoring_parts(verbose=False)
            print(f"Fisher scoring parts (it. {self.num_iterations}): {fisher_scoring_parts}")
            return fisher_scoring_parts
        else:
            return aggregator_results[0]


class FederatedLogisticRegression(StarAggregator):
    def __init__(self, flame):
        """
        Initializes aggregator object and iteratively checks for convergence
        and aggegates fisher scoring parts from each node.
        """
        super().__init__(flame)  # Connects this aggregator to the FLAME components
        self.glm = FederatedGLM()

        # additional tmp flag to keep track of convergence *independent* of convergence in has_converged() to modify final result
        self._convergence_flag = False

    def aggregation_method(self, analysis_results):
        """
        Aggregates the results received from all analyzer nodes.

        :param analysis_results: A list of analysis results from each analyzer node.
        :return: The aggregated result (e.g., total patient count across all analyzers).
        """
        print()
        print(f"------ AGGREGATION in {self.id} ------")
        if not self._convergence_flag:
            self.glm.set_node_results(analysis_results)
            self.glm.aggregate_results()

            print(f"Aggregated coefficients (it. {self.num_iterations}): {self.glm.get_coefs()}")
            print(f"Aggregated convergence flag (it. {self.num_iterations}): {self._convergence_flag}")

            return self.glm.get_coefs(), self._convergence_flag
        else:

            print(f"Final coefficients (it. {self.num_iterations}): {self.glm.get_coefs()}")

            return self.glm.get_summary()

    def has_converged(self, result, last_result):
        """
        Determines if the aggregation process has converged.

        :param result: The current aggregated result.
        :param last_result: The aggregated result from the previous iteration.
        :param num_iterations: The number of iterations completed so far.
        :return: True if the aggregation has converged; False to continue iterations.
        """
        print()
        print("------ CONVERGENCE CHECK ------")
        if self._convergence_flag:
            print(f"Converged after {self.num_iterations} iterations.")
            return True

        if last_result is None:
            print("First iteration, no convergence check.")
            return False

        diff = np.linalg.norm(result[0] - last_result[0], ord=2).item()
        print(f"Difference between coefficients (it. {self.num_iterations}): {diff}")

        convergence = self.glm.check_convergence(last_result[0], result[0], tol=1e-4)
        if convergence:
            # TODO: Currently, a the following is a workaround. Another round of analysis is done with no results such that
            # the final result can be modified. Maybe there is a better solution in the future.
            self._convergence_flag = True
            return False  # here, False is returned even though convergence is achieved to perform a final round, where a summary is returned instead of coefficients.
        elif self.num_iterations > 100:
            # TODO: Include option for max iteration and not hardcoded tol
            print(
                "Maximum number of 100 iterations reached. Returning current results."
            )
            return True
        else:
            return False


def main():
    """
    Sets up and initiates the distributed analysis using the FLAME components.

    - Defines the custom analyzer and aggregator classes.
    - Specifies the type of data and queries to execute.
    - Configures analysis parameters like iteration behavior and output format.
    """
    seed = 42
    X, y = simulate_logistic_regression(
        seed, n=50, p=3
    )

    data_splits = list(zip(X, y))  # create 5 splits of the data for 5 analyzer nodes


    StarModelTester(
        data_splits=data_splits,
        analyzer=LocalFisherScoring,  # Custom analyzer class (must inherit from StarAnalyzer)
        aggregator=FederatedLogisticRegression,  # Custom aggregator class (must inherit from StarAggregator)
        data_type="s3",  # Type of data source ('fhir' or 's3')
        simple_analysis=False,  # True for single-iteration; False for multi-iterative analysis
        output_type="str",  # Output format for the final result ('str', 'bytes', or 'pickle')
        analyzer_kwargs=None,  # Additional keyword arguments for the custom analyzer constructor (i.e. MyAnalyzer)
        aggregator_kwargs=None,  # Additional keyword arguments for the custom aggregator constructor (i.e. MyAggregator)
    )


if __name__ == "__main__":
    main()
