import numpy as np
from fedstats.aggregation.aggregator import Aggregator

class AverageAggregator(Aggregator):
    def __init__(self, results: list) -> None:
        """
        Wrapper class to handle aggregation via Average

        The object is either a AverageAggregatorUnit or a AverageAggregatorCollection. For typing details, see the two classes.

        :param results: A list of tuples or a list of lists of tuples. See subclasses for more.
        """

        super().__init__(results)
        self.isunit = self.check_input()
        aggregator = AverageAggregatorUnit if self.isunit else AverageAggregatorCollection
        self.aggregator = aggregator(results)

    def check_input(self) -> bool:
        if all(isinstance(item, tuple) for item in self.results):
            return True 
        elif all(isinstance(item, list) and all(isinstance(subitem, tuple) for subitem in item) for item in self.results):
            return False
        else:
            raise TypeError("Input should be either a list of tuples, or a list of list of tuples.")


    def aggregate_results(self) -> None:
        self.aggregator.aggregate_results()


    def get_results(self) -> dict[str, np.ndarray] | dict[str, float | tuple[float, float]]:
        return self.aggregator.get_results()




class AverageAggregatorUnit:
    def __init__(self, results: list[tuple[float, int]]) -> None:
        """
        AverageAggregator can be used to aggregate K single effect sizes. Either weighted by sample size.

        :param results: A list containing K tuples of length 2. First element is the effect size, the second the sample size,
        """
        self.results = results
        self.K = len(results)
        self.effect_sizes, self.n_samples = map(lambda x: np.array(x), zip(*results))
        self.weights = self.n_samples / self.n_samples.sum()


    def calculate_pooled_effect_size(self) -> None:
        """
        Calculate the pooled effect size (weighted mean).
        """
        self.pooled_effect_size = (self.effect_sizes * self.weights).sum().item()


    def aggregate_results(self) -> None:
        """
        Perform the meta-analysis and return all results.

        :param calculate_heterogeneity: Boolean flag whether q statisc should be calculated. 
        """
        self.calculate_pooled_effect_size()
        # TODO: self.calculate_confidence_interval()   --> search literature for method or just use CLT properties if nothing is there


    def get_results(self) -> dict[str, tuple[float, float] | float]:
        """
        Get results fom the object

        Returns:
            A dict with results.
        """
        results = {}
        results["aggregated_results"] = self.pooled_effect_size
        # TODO: results["confidence_interval"] = (self.ci_lower, self.ci_upper)
        return results





class AverageAggregatorCollection:
    def __init__(self, results: list[list[tuple[float, int]]]):
        """
        AverageAggregatorCollection is a wrapper of AverageAggregatorUnit for multiple effect sizes.

        This class receives and processes estimators from $K$ servers, where each server produces a list of results.
        Each element in these lists is itself a list containing $P$ tuples, corresponding to $P$ effect sizes (effect size, number of local samples).
        In a standard scenario, all n would be the same. However, it is designed in this way to provide flexibility.

        :param results: A list of length K containting p lists with tuples of length 2. First element is the effect size, the second the variance,

        Example how the results from K=3 servers with P=2 effect sizes may look like. E.g. mu32 corresponds to the effect size of the 2nd estimator of sever 3.
            >>> data = [
            ...     [(mu11, n11), (mu12, n12)],  # Results from server 1
            ...     [(mu21, n21), (mu22, n22)],  # Results from server 2
            ...     [(mu31, n31), (mu32, n32)]   # Results from server 3
            ... ]

        """
        self.results = results
        self.K = len(results)

        self.aggregator_units= [(AverageAggregatorUnit([est[p] for est in results])) for p in range(len(results[0]))]


    def calculate_pooled_effect_size(self) -> None:
        """
        Calculate the pooled effect size (weighted mean).
        """
        for unit in self.aggregator_units:
            unit.calculate_pooled_effect_size()


    def aggregate_results(self) -> None:
        """
        Perform the meta-analysis and return all results.

        :param calculate_heterogeneity: Boolean flag whether q statisc should be calculated. 
        """
        self.calculate_pooled_effect_size()


    def get_results(self) -> dict[str, np.ndarray]:
        """
        Get results fom the object
        """
        return dict(aggregated_results = np.array([unit.pooled_effect_size for unit in self.aggregator_units]))




