"""Classes for meta analysis aggregation."""

from collections.abc import Mapping

import numpy as np
from scipy.stats import norm

from fedstats.aggregation.aggregator import Aggregator


class MetaAnalysisAggregator(Aggregator):
    """Wrapper class to handle aggregation via meta analysis."""

    node_results: list[tuple[float, float]] | list[list[tuple[float, float]]]

    def __init__(self, node_results: list) -> None:
        """Initialize the MetaAnalysisAggregator.

        Depending on the input, the object is either a MetaAnalysisAggregatorUnit or a MetaAnalysisAggregatorCollection.
        For typing details, see the two respective classes.

        Arguments:
            node_results
                A list of tuples or a list of lists of tuples.
                For MetaAnalysisAggregatorUnit: A list containing K tuples of length 2.
                For MetaAnalysisAggregatorCollection: A list of length K containing P lists with tuples of length 2.
                First element is the effect size, the second the variance.
        """
        super().__init__(node_results)
        self.isunit = self.check_input()
        aggregator = MetaAnalysisAggregatorUnit if self.isunit else MetaAnalysisAggregatorCollection
        self.aggregator = aggregator(node_results)

    def check_input(self) -> bool:
        """
        Check the input format to determine if it's for a unit or collection aggregator.

        Returns
        -------
            True if input is for MetaAnalysisAggregatorUnit, False if input is for MetaAnalysisAggregator
            Collection.

        Raises
        ------
            TypeError: If the input format is invalid.
            ValueError: If the input list is empty.
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

    def aggregate_results(self, calculate_heterogeneity: bool = False) -> None:
        """
        Perform the aggregation and store results in the object.

        Parameters
        ----------
            calculate_heterogeneity : bool, optional
                Boolean flag whether q statistic should be calculated. Default is False.
        """
        self.aggregator.aggregate_results(calculate_heterogeneity=calculate_heterogeneity)

    def get_aggregated_results(self) -> Mapping[str, np.ndarray | float | tuple[float, float]]:
        """
        Get aggregated results from the object.

        Returns
        -------
            A dict with aggregated results.
                aggregated_results: The pooled effect size(s).
                aggregated_variance: The variance of the pooled effect size(s).
                confidence_interval: Lower and upper bound of the confidence interval for the pooled effect size(s).
                q_statistic: If calculated, the Q-statistic for heterogeneity.
        """
        return self.aggregator.get_aggregated_results()


class MetaAnalysisAggregatorUnit:
    """
    MetaAnalysisAggregatorUnit can aggregate effect sizes from K clients.

    The method used here is a fixed effect meta analysis also called inverse variance weights.
    See for example Table 1 right column in https://doi.org/10.1093/bioinformatics/btq340
    """

    def __init__(self, node_results: list[tuple[float, float]]) -> None:
        """
        Initialize the MetaAnalysisAggregatorUnit.

        Parameters
        ----------
            node_results
                A list containing K tuples of length 2.
                First element is the effect size, the second the variance.

        Raises
        ------
            ValueError: If the input list is empty.
        """
        if not node_results:
            raise ValueError("Node results list is empty. Please provide valid node results.")

        self.node_results = node_results
        self.K = len(node_results)
        self.effect_sizes, self.variances = map(lambda x: np.array(x), zip(*node_results, strict=False))
        self.weights = 1 / self.variances

    def calculate_pooled_effect_size(self) -> None:
        """Calculate the pooled effect size (weighted mean)."""
        self.pooled_effect_size = (np.sum(self.weights * self.effect_sizes) / np.sum(self.weights)).item()

    def calculate_pooled_variance(self) -> None:
        """Calculate the variance of the pooled effect size."""
        self.pooled_variance = 1 / np.sum(self.weights).item()

    def calculate_confidence_interval(self, alpha_level: float = 0.05) -> None:
        r"""
        Calculate a symmetric $(1-\alpha)$ confidence interval for the pooled effect size.

        Parameters
        ----------
            alpha_level : float, optional
                A float in [0,1] that representes the confidence level. Default is 0.05 for a 95% confidence interval.

        Raises
        ------
            ValueError: If alpha_level is not in [0,1].
        """
        if not (0 <= alpha_level <= 1):
            raise ValueError("alpha_level should be a float in the interval [0,1].")

        std_error = np.sqrt(self.pooled_variance)
        quantile = norm.ppf(1 - alpha_level / 2, loc=0, scale=1)
        self.ci_lower = (self.pooled_effect_size - quantile * std_error).item()
        self.ci_upper = (self.pooled_effect_size + quantile * std_error).item()

    def calculate_q_statistic(self) -> None:
        """Calculate the Q-statistic for heterogeneity."""
        self.q_stat = np.sum(self.weights * (self.effect_sizes - self.pooled_effect_size) ** 2).item()

    def aggregate_results(self, calculate_heterogeneity: bool = False) -> None:
        """
        Perform the meta analysis and store results in the object.

        Parameters
        ----------
            calculate_heterogeneity : bool, optional
                Boolean flag whether q statistic should be calculated.
        """
        self.calculate_pooled_effect_size()
        self.calculate_pooled_variance()

        self.calculate_confidence_interval()

        if calculate_heterogeneity:
            self.calculate_q_statistic()

    def get_aggregated_results(self) -> dict[str, tuple[float, float] | float]:
        """
        Get results fom the object.

        Returns
        -------
            A dict with results.
                aggregated_results: The pooled effect size(s).
                aggregated_variance: The variance of the pooled effect size(s).
                confidence_interval: Lower and upper bound of the confidence interval for the pooled effect size(s).
                q_statistic: If calculated, the Q-statistic for heterogeneity.

        """
        results = {}
        results["aggregated_results"] = self.pooled_effect_size
        results["aggregated_variance"] = self.pooled_variance
        results["confidence_interval"] = (self.ci_lower, self.ci_upper)
        if getattr(self, "q_stat", None) is not None:
            results["q_statistic"] = self.q_stat
        return results


class MetaAnalysisAggregatorCollection:
    """
    MetaAnalysisAggregatorCollection can be used to aggregate K lists of P effect sizes, weighted by variance.

    MetaAnalysisAggregatorCollection is a wrapper of MetaAnalysisAggregatorUnit for multiple effect sizes.

    This class receives and processes estimators from $K$ servers, where each server produces a list of results.
    Each element in these lists is itself a list containing $P$ tuples,
    corresponding to $P$ effect sizes (effect size, variance)
    """

    def __init__(self, node_results: list[list[tuple[float, float]]]):
        """Initialize the MetaAnalysisAggregatorCollection by creating P MetaAnalysisAggregatorUnits.

        Parameters
        ----------
            node_results
                A list of length K containing P lists with tuples of length 2.
                First element is the effect size, the second the variance.

        Examples
        --------
            Example how the results from K=3 servers with P=2 effect sizes may look like.
            Here, mu32 corresponds to the effect size of the 2nd estimator of sever 3.
                >>> data = [
                ...     [(mu11, var11), (mu12, var12)],  # Results from server 1
                ...     [(mu21, var21), (mu22, var22)],  # Results from server 2
                ...     [(mu31, var31), (mu32, var32)]   # Results from server 3
                ... ]

        Raises
        ------
            TypeError: If the input format is invalid.
            ValueError: If the input list is empty.
        """
        if not node_results:
            raise ValueError("Node results list is empty. Please provide valid node results.")

        self.node_results = node_results
        self.K = len(node_results)

        self.aggregator_units = [
            (MetaAnalysisAggregatorUnit([est[p] for est in node_results])) for p in range(len(node_results[0]))
        ]

    def calculate_pooled_effect_size(self) -> None:
        """Calculate the pooled effect size (weighted mean)."""
        for unit in self.aggregator_units:
            unit.calculate_pooled_effect_size()

    def calculate_pooled_variance(self) -> None:
        """Calculate the variance of the pooled effect size."""
        for unit in self.aggregator_units:
            unit.calculate_pooled_variance()

    def calculate_confidence_interval(self, alpha_level: float = 0.05) -> None:
        r"""
        Calculate a symmetric $(1-\alpha)$% confidence interval for the pooled effect size.

        Parameters
        ----------
            alpha_level : float, optional
                A float in [0,1] that representes the confidence level. Default is 0.05 for a 95% confidence interval.

        Raises
        ------
            ValueError: If alpha_level is not in [0,1].
        """
        if not (0 <= alpha_level <= 1):
            raise ValueError("alpha_level should be a float in the interval [0,1].")

        for unit in self.aggregator_units:
            unit.calculate_confidence_interval(alpha_level=alpha_level)

    def calculate_q_statistic(self) -> None:
        """Calculate the Q-statistic for heterogeneity."""
        self.q_stat_calculated = True
        for unit in self.aggregator_units:
            unit.calculate_q_statistic()

    def aggregate_results(self, calculate_heterogeneity: bool = False) -> None:
        """
        Perform the meta analysis and store results in the object.

        Parameters
        ----------
            calculate_heterogeneity : bool, optional
                Boolean flag whether q statistic should be calculated. Default is False.
        """
        self.calculate_pooled_effect_size()
        self.calculate_pooled_variance()
        self.calculate_confidence_interval()

        if calculate_heterogeneity:
            self.calculate_q_statistic()

    def get_aggregated_results(self) -> dict[str, np.ndarray]:
        """
        Get aggregated results from the object.

        Returns
        -------
            A dict with aggregated results.
                aggregated_results: The pooled effect size(s) stored as a numpy array of length P.
                aggregated_variance: The variance of the pooled effect size(s) stored as a numpy array of length P.
                confidence_interval: Lower and upper bound of the confidence interval for the pooled effect size(s)
                    stored as a numpy array of size Px2, with rows corresponding to (ci_lower, ci_upper).
                q_statistic: If calculated, the Q-statistic for heterogeneity stored as a numpy array of length P
        """
        results = {}
        results["aggregated_results"] = np.array([unit.pooled_effect_size for unit in self.aggregator_units])
        results["aggregated_variance"] = np.array([unit.pooled_variance for unit in self.aggregator_units])
        results["confidence_interval"] = np.array([(unit.ci_lower, unit.ci_upper) for unit in self.aggregator_units])
        if getattr(self, "q_stat_calculated", None) is not None:
            results["q_statistic"] = np.array([unit.q_stat for unit in self.aggregator_units])
        return results
