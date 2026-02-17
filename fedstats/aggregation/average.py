"""Aggregation via averaging."""

import numpy as np

from fedstats.aggregation.aggregator import Aggregator


class AverageAggregator(Aggregator):
    """Wrapper class to handle aggregation via averaging."""

    def __init__(self, node_results: list) -> None:
        """Initialize the AverageAggregator.

        Depending on the input, the object is either a AverageAggregatorUnit or a AverageAggregatorCollection.
        For typing details, see the two respective classes.

        Parameters
        ----------
        node_results
            A list of tuples or a list of lists of tuples.
            For AverageAggregatorUnit: A list containing K tuples of length 2.
            For AverageAggregatorCollection: A list of length K containing P lists with tuples of length 2.
            In the tuple, the first element is the effect size, the second the number of local samples.
        """
        super().__init__(node_results)
        self.isunit = self.check_input()
        aggregator = AverageAggregatorUnit if self.isunit else AverageAggregatorCollection
        self.aggregator = aggregator(node_results)

    def check_input(self) -> bool:
        """
        Check the input format to determine if it's for a unit or collection aggregator.

        Returns
        -------
            True if input is for AverageAggregatorUnit, False if input is for AverageAggregatorCollection.

        Raises
        ------
            ValueError: If the input list is empty.
            TypeError: If the input format is invalid.
        """
        if not self.node_results:
            raise ValueError("Node results list is empty. Please provide valid node results.")

        if all(isinstance(item, tuple) for item in self.node_results):
            return True
        elif all(
            isinstance(item, list) and all(isinstance(subitem, tuple) for subitem in item) for item in self.node_results
        ):
            return False
        else:
            raise TypeError("Input should be either a list of tuples, or a list of list of tuples.")

    def aggregate_results(self) -> None:
        """Perform the aggregation and store results in the object."""
        self.aggregator.aggregate_results()

    def get_aggregated_results(self) -> dict[str, np.ndarray] | dict[str, float] | dict[str, tuple[float, float]]:
        """
        Get aggregated results from the object.

        Returns
        -------
            A dict with aggregated results.
            The format of the dict depends on the format of the input and the type of aggregator used.
            The pooled effect size(s) is/are stored under the key "aggregated_results".
        """
        return self.aggregator.get_results()


class AverageAggregatorUnit:
    """AverageAggregatorUnit can be used to aggregate K single effect sizes, weighted by sample size."""

    def __init__(self, node_results: list[tuple[float, int]]) -> None:
        """Initialize the AverageAggregatorUnit.

        Parameters
        ----------
            node_results
                A list containing K tuples of length 2.
                First element is the effect size, the second the sample size.

        Raises
        ------
            ValueError: If the input list is empty.
        """
        if not node_results:
            raise ValueError("Node results list is empty. Please provide valid node results.")

        self.node_results = node_results
        self.K = len(node_results)
        self.effect_sizes, self.n_samples = map(lambda x: np.array(x), zip(*node_results, strict=False))
        self.weights = self.n_samples / self.n_samples.sum()

    def calculate_pooled_effect_size(self) -> None:
        """Calculate the pooled effect size (weighted mean) and store it in the object."""
        self.pooled_effect_size = (self.effect_sizes * self.weights).sum().item()

    def aggregate_results(self) -> None:
        """Perform the aggregation and store results in the object."""
        self.calculate_pooled_effect_size()
        # TODO: self.calculate_confidence_interval()
        # --> search literature for method or just use CLT properties if nothing is there
        # Good idea: make basically a fedAvg formula for the variance,
        #   i.e. something like instead of 1/(n-1) \sum (x_i-mu_x) with appripriate weights n_s/n
        # Also look here (attention! qestion is for a SEQUENCE a): https://math.stackexchange.com/questions/3135950/central-limit-theorem-for-weighted-average

    def get_results(self) -> dict[str, float] | dict[str, tuple[float, float]]:
        """Get results fom the object.

        Returns
        -------
            A dict with results.
            The pooled effect size is stored under the key "aggregated_results".

        """
        results = {}
        results["aggregated_results"] = self.pooled_effect_size
        # TODO: results["confidence_interval"] = (self.ci_lower, self.ci_upper)
        return results


class AverageAggregatorCollection:
    """AverageAggregatorCollection can be used to aggregate K lists of P effect sizes, weighted by sample size."""

    def __init__(self, node_results: list[list[tuple[float, int]]]):
        """Initialize the AverageAggregatorCollection by creating AverageAggregatorUnits for each of the P effect sizes.

        This class receives and processes estimators from $K$ servers, where each server produces a list of results.
        Each element in these lists is itself a list containing $P$ tuples,
        corresponding to $P$ effect sizes (effect size, number of local samples).
        Typically, the number of local samples is the same for all effect sizes of a server, but this is not required.

        Parameters
        ----------
            node_results
                A list of length K containing P lists with tuples of length 2.
                First element is the effect size, the second the number of local samples.

        Examples
        --------
            Input data from K=3 servers with P=2 effect sizes may look like this:
            Here, mu32 corresponds to the effect size of the 2nd estimator of sever 3,
            and n31 corresponds to the number of local samples used to calculate mu31.
            >>> data = [
            ...     [(mu11, n11), (mu12, n12)],  # Results from server 1
            ...     [(mu21, n21), (mu22, n22)],  # Results from server 2
            ...     [(mu31, n31), (mu32, n32)]   # Results from server 3
            ... ]

        """
        if not node_results:
            raise ValueError("Node results list is empty. Please provide valid node results.")

        self.node_results = node_results
        self.K = len(node_results)

        self.aggregator_units = [
            (AverageAggregatorUnit([est[p] for est in node_results])) for p in range(len(node_results[0]))
        ]

    def calculate_pooled_effect_size(self) -> None:
        """Calculate the pooled effect size as a weighted mean and store it in the object."""
        for unit in self.aggregator_units:
            unit.calculate_pooled_effect_size()

    def aggregate_results(self) -> None:
        """Aggregate results by calculating the pooled effect sizes and store them in the object."""
        self.calculate_pooled_effect_size()

    def get_results(self) -> dict[str, np.ndarray]:
        """Get the pooled effect sizes from the object.

        Returns
        -------
            A dict with results.
            The pooled effect sizes are stored under the key "aggregated_results" as a numpy array of length P.
        """
        return dict(aggregated_results=np.array([unit.pooled_effect_size for unit in self.aggregator_units]))
