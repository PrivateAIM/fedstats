"""Aggregator for combining p-values using Fisher's method."""

import numpy as np
from scipy.stats import combine_pvalues, norm

from .aggregator import Aggregator


class FisherAggregator(Aggregator):
    """Aggregator for combining p-values using Fisher's method.

    This class takes local (estimate, stddev) tuples from each site, converts them to p-values, and combines them using
    Fisher's method. The final result is a single combined p-value.
    This method assumes that the underlying distributions of the estimates are approximately normal.
    """

    node_results: list[tuple[float, float]]

    def __init__(self, node_results: list[tuple[float, float]]):
        """Initialize the FisherAggregator with local results.

        Parameters
        ----------
             node_results
                A list of tuples, where each tuple contains (estimate, stddev).
        """
        super().__init__(node_results)

    def aggregate_results(self, verbose: bool = False) -> None:
        """
        Compute a single combined p-value for the given local results using Fisher's method.

        Parameters
        ----------
            verbose
                If True, print the local p-values for each site during aggregation.

        Raises
        ------
            ValueError: If node_results is None or empty.
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
        Return the combined p-value after aggregation.

        Returns
        -------
            The combined p-value as a float.

        Raises
        ------
            ValueError: If aggregate_results has not been called yet.
        """
        if hasattr(self, "combined_p_value"):
            return self.combined_p_value
        else:
            raise ValueError("No aggregated result computed. Call aggregate_results first.")

    @staticmethod
    def _estimate_to_pvalue(estimate: float | np.ndarray, stddev: float | np.ndarray) -> float | np.ndarray:
        """Convert estimate/stddev to a two-sided p-value using z-scores.

        Works for both scalars and numpy arrays.

        Returns nan for any cases where stddev is zero, as the test is not valid in those cases.

        Parameters
        ----------
            estimate
                The local estimate(s) from the site(s).
            stddev
                The local standard deviation(s) from the site(s).

        Returns
        -------
            The corresponding p-value(s) as a float or numpy array.
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
                "Standard deviation of zero encountered in _estimate_to_pvalue; returning NaN for those p-values.",
                stacklevel=2,
            )

        # If the result is a single element, return it as a scalar
        return p_val.item() if p_val.size == 1 else p_val
