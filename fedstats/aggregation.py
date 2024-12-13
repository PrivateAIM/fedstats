from typing import Dict
import numpy as np
from scipy.stats import norm

class MetaAnalysisAggregatorUnit:
    def __init__(self, results: list[tuple[float, float]]):
        """
        MetaAnalysisAggregator can be used to aggregate K single effect sizes.

        Used method is a fixed effect meta analysis or also called inverse variance weights. 
        See for example Table 1 right column in https://doi.org/10.1093/bioinformatics/btq340

        :param results: A list containing K tuples of length 2. First element is the effect size, the second the variance,
        """
        self.results = results
        self.K = len(results)
        self.effect_sizes, self.variances = map(lambda x: np.array(x), zip(*results))
        self.weights = 1 / self.variances


    def calculate_pooled_effect_size(self) -> None:
        """
        Calculate the pooled effect size (weighted mean).
        """
        self.pooled_effect_size = (np.sum(self.weights * self.effect_sizes) / np.sum(self.weights)).item()


    def calculate_pooled_variance(self) -> None:
        """
        Calculate the variance of the pooled effect size.
        """
        self.pooled_variance = 1 / np.sum(self.weights).item()


    def calculate_confidence_interval(self, alpha_level: float = 0.05) -> None:
        """
        Calculate a symmetric $(1-\alpha)$% confidence interval for the pooled effect size.

        Parameters:
            z_score (float): Z-score for the desired confidence level (default is 1.96 for 95%).

        """
        std_error = np.sqrt(self.pooled_variance)
        quantile = norm.ppf(1-alpha_level/2, loc=0, scale=1)
        self.ci_lower = (self.pooled_effect_size - quantile * std_error).item()
        self.ci_upper = (self.pooled_effect_size + quantile * std_error).item()


    def calculate_q_statistic(self) -> None:
        """
        Calculate the Q-statistic for heterogeneity.
        """
        self.q_stat = np.sum(self.weights * (self.effect_sizes - self.pooled_effect_size) ** 2).item()


    def aggregate_results(self, calculate_heterogeneity: bool = False):
        """
        Perform the meta-analysis and return all results.

        Returns:
            dict: Contains pooled effect size, confidence interval, and Q-statistic.
        """
        
        self.calculate_pooled_effect_size()
        self.calculate_pooled_variance()
        self.calculate_confidence_interval()
    
        if calculate_heterogeneity:
            self.calculate_q_statistic()


    def get_results(self) -> Dict:
        """
        Get results fom the object
        """
        results = {}
        results["aggregated_results"] = self.pooled_effect_size
        results["aggregated_variance"] = self.pooled_variance
        results["confidence_interval"] = (self.ci_lower, self.ci_upper)
        if getattr(self, "q_stat", None) is not None:
            results['q_statistic'] = self.q_stat
        return results






class MetaAnalysisAggregatorCollection:
    def __init__(self, results: list[list[tuple[float, float]]]):
        """
        MetaAnalysisAggregatorCollection is a wrapper of MetaAnalysisAggregatorUnit for multiple effect sizes.

        This class receives and processes estimators from `K` servers, where each server produces a list of results.
        Each element in these lists is itself a list containing `P` tuples, corresponding to `P` effect sizes (effect size, variance)

        :param results: A list of length K containting p lists with tuples of length 2. First element is the effect size, the second the variance,

        Example how the results from K=3 servers with P=2 effect sizes may look like. E.g. mu32 corresponds to the effect size of the 2nd estimator of sever 3.
            >>> data = [
            ...     [(mu11, var11), (mu12, var12)],  # Results from server 1
            ...     [(mu21, var21), (mu22, var22)],  # Results from server 2
            ...     [(mu31, var31), (mu32, var32)]   # Results from server 3
            ... ]

        """
        self.results = results
        self.K = len(results)

        self.aggregator_units= [(MetaAnalysisAggregatorUnit([est[p] for est in results])) for p in range(len(results[0]))]


    def calculate_pooled_effect_size(self) -> None:
        """
        Calculate the pooled effect size (weighted mean).
        """
        for unit in self.aggregator_units:
            unit.calculate_pooled_effect_size()


    def calculate_pooled_variance(self) -> None:
        """
        Calculate the variance of the pooled effect size.
        """
        for unit in self.aggregator_units:
            unit.calculate_pooled_variance()


    def calculate_confidence_interval(self, alpha_level: float = 0.05) -> None:
        """
        Calculate a symmetric $(1-\alpha)$% confidence interval for the pooled effect size.

        Parameters:
            z_score (float): Z-score for the desired confidence level (default is 1.96 for 95%).

        Returns:
            (float, float): Lower and upper bounds of the confidence interval.
        """
        for unit in self.aggregator_units:
            unit.calculate_confidence_interval(alpha_level=alpha_level)


    def calculate_q_statistic(self) -> None:
        """
        Calculate the Q-statistic for heterogeneity.
        """
        for unit in self.aggregator_units:
            unit.calculate_q_statistic()


    def aggregate_results(self, calculate_heterogeneity: bool = False):
        """
        Perform the meta-analysis and return all results.

        Returns:
            dict: Contains pooled effect size, confidence interval, and Q-statistic.
        """
        
        self.calculate_pooled_effect_size()
        self.calculate_pooled_variance()
        self.calculate_confidence_interval()
    
        if calculate_heterogeneity:
            self.calculate_q_statistic()


    def get_results(self) -> Dict:
        """
        Get results fom the object
        """
        results = {}
        results['aggregated_results'] = np.array([unit.pooled_effect_size for unit in self.aggregator_units])
        results['aggregated_variance'] = np.array([unit.pooled_variance for unit in self.aggregator_units])
        results['confidence_interval'] = np.array([(unit.ci_lower, unit.ci_upper) for unit in self.aggregator_units])
        if getattr(self, "q_stat", None) is not None:
            results['q_statistic'] = np.array([unit.q_stat for unit in self.aggregator_units])
        return results




