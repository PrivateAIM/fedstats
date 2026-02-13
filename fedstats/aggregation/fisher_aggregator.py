import numpy as np
from scipy.stats import combine_pvalues, norm

from .aggregator import Aggregator


class FisherAggregator(Aggregator):
    """
    Aggregates local (estimate, stddev) results by converting them to p-values
    and combining with Fisher's method in the scipy implementation.
    The final result is a single combined p-value.
    This requires that the underlying distributions of the estimates are approximately normal.

    :param node_results: A list of tuples (estimate, stddev) from each site.
    """

    node_results: list[tuple[float, float]]

    def __init__(self, node_results: list[tuple[float, float]]):
        super().__init__(node_results)

    def aggregate_results(self, verbose: bool = False) -> None:
        """
        Computes a single combined p-value.

        :param verbose: If True, prints local p-values for each site during aggregation.
        """
        if not self.node_results:
            raise ValueError("No results to aggregate.")

        p_values = []
        for i, (est, sd) in enumerate(self.node_results):
            p_val = np.clip(self._estimate_to_pvalue(est, sd), 1e-16, 1.0)
            if verbose:
                print(f"Site {i} local p-value: {p_val}")
            p_values.append(p_val)
        _, combined_p_value = combine_pvalues(p_values, method="fisher")
        self.combined_p_value: float = combined_p_value  # type: ignore - scipy does not specify the return type correctly

    def get_aggregated_results(self) -> float:
        """
        Returns the stored aggregated result.

        :return: The combined p-value from the aggregation.
        """
        if hasattr(self, "combined_p_value"):
            return self.combined_p_value
        else:
            raise ValueError("No aggregated result computed. Call aggregate_results first.")

    @staticmethod
    def _estimate_to_pvalue(estimate: float | np.ndarray, stddev: float | np.ndarray) -> float | np.ndarray:
        """
        Convert estimate/stddev to a two-sided p-value using z-scores.
        Works for both scalars and numpy arrays.

        Returns nan for any cases where stddev is zero, as the test is not valid in those cases.

        :param estimate: The local estimate(s) from the site(s).
        :param stddev: The local standard deviation(s) from the site(s).
        :return: The corresponding p-value(s).
        """
        # Ensure inputs are numpy arrays
        estimate = np.asarray(estimate)
        stddev = np.asarray(stddev)

        # Avoid division by zero: temporarily set zeros to nan
        safe_stddev = np.where(stddev == 0, np.nan, stddev)

        # Compute z-scores elementwise
        z_score = estimate / safe_stddev

        # Compute two-sided p-values
        p_val = 2.0 * (1.0 - norm.cdf(np.abs(z_score)))

        # Wherever stddev was zero, return nan as the test is not valid
        p_val = np.where(np.isnan(z_score), np.nan, p_val)

        # Log a warning if any stddev were zero
        if np.any(stddev == 0):
            import warnings

            warnings.warn(
                "Standard deviation of zero encountered in _estimate_to_pvalue; returning NaN for those p-values."
            )

        # If the result is a single element, return it as a scalar
        return p_val.item() if p_val.size == 1 else p_val
