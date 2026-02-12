from .aggregator import Aggregator
from scipy.stats import combine_pvalues, norm

import numpy as np

class FisherAggregator(Aggregator):
    """
    Aggregates local (estimate, stddev) results by converting them to p-values
    and combining with Fisher's method in the scipy implementation.
    The final result is a single combined p-value.

    :param results: A list of tuples (estimate, stddev) from each site.
    """
    def __init__(self, node_results: list[tuple]):
        super().__init__(node_results)

    def aggregate_results(self, verbose=False):
        """
        Computes a single combined p-value.
        """
        if not self.node_results:
            raise ValueError("No results to aggregate.")

        p_values = []
        for i, (est, sd) in enumerate(self.node_results):
            p_val = np.clip(self._estimate_to_pvalue(est, sd), 1e-16, 1.0)
            if verbose:
                print(f"Site {i} local p-value: {p_val}")
            p_values.append(p_val)
        _, combined_p_value = combine_pvalues(p_values, method='fisher')
        self.combined_p_value = combined_p_value


    def get_aggregated_results(self) -> float:
        """
        Returns the stored aggregated result.
        """
        if hasattr(self, 'combined_p_value'):
            return self.combined_p_value
        else:
            raise ValueError("No aggregated result computed. Call aggregate_results first.")

    @staticmethod
    def _estimate_to_pvalue(estimate, stddev):
        """
        Convert estimate/stddev to a two-sided p-value using z-scores.
        Works for both scalars and numpy arrays.
        """
        import numpy as np

        # Ensure inputs are numpy arrays
        estimate = np.asarray(estimate)
        stddev = np.asarray(stddev)

        # Avoid division by zero: temporarily set zeros to nan
        safe_stddev = np.where(stddev == 0, np.nan, stddev)

        # Compute z-scores elementwise
        z_score = estimate / safe_stddev

        # Compute two-sided p-values
        p_val = 2.0 * (1.0 - norm.cdf(np.abs(z_score)))

        # Wherever stddev was zero, override p-value to 1.0
        p_val = np.where(np.isnan(z_score), 1.0, p_val)

        # If the result is a single element, return it as a scalar
        return p_val.item() if p_val.size==1 else p_val

