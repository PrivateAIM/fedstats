"""Aggregation for Generalized Linear Model fisher scoring results."""

from functools import reduce

import numpy as np
from scipy.stats import norm

from fedstats.aggregation.aggregator import Aggregator


class FedGLM(Aggregator):
    """
    Aggregator for Generalized Linear Model fisher scoring results.

    The idea for the aggregation is based on: https://www.mdpi.com/1999-4893/15/7/243
    """

    info_calculated: bool = False

    def __init__(self, node_results: list | None = None) -> None:
        """
        Initialize the FedGLM aggregator.

        Parameters
        ----------
             node_results
                A list of tuples, where each tuple contains two np.arrays.
                The first array is the fisher information matrix,
                the second array is the right hand side of the fisher scoring update.
        """
        super().__init__(node_results)
        self.coefs = np.array(np.inf)  # TODO: Make a better solution for init
        self.iter = 0

    def set_node_results(self, node_results: list[tuple[np.ndarray, np.ndarray]]) -> None:
        """
        Set node results for the aggregator.

        Parameters
        ----------
            node_results
                A list of tuples, where each tuple contains two np.arrays.
                The first array is the fisher information matrix,
                the second array is the right hand side of the fisher scoring update.
        """
        self.node_results = node_results

    def aggregate_results(self, calc_info: bool = False) -> None:
        """
        Perform the aggregation and store results in the object.

        Parameters
        ----------
            calc_info : bool, optional
                Boolean flag whether to calculate standard errors, z-scores and p-values after aggregation.
        """
        if self.node_results is None:
            raise ValueError("No node results to aggregate. Please set node results first.")

        if len(self.node_results) == 0:
            raise ValueError("Node results list is empty. Please provide valid node results.")

        fisher_infos = [res[0] for res in self.node_results]
        rhss = [res[1] for res in self.node_results]

        self.fisher_info_agg = reduce(lambda x, y: x + y, fisher_infos)
        self.rhs_agg = reduce(lambda x, y: x + y, rhss)
        try:
            coefs = np.linalg.solve(self.fisher_info_agg, self.rhs_agg)
        except np.linalg.LinAlgError:
            coefs = np.linalg.pinv(self.fisher_info_agg) @ self.rhs_agg
        self.coefs = coefs
        self.iter += 1

        if calc_info:
            self.calc_info()

    def check_convergence(self, coefs_old: np.ndarray, coefs_new: np.ndarray, tol: float = 1e-6) -> bool:
        """
        Check convergence of the fisher scoring algorithm.

        Parameters
        ----------
            coefs_old
                Coefficients from the previous iteration.
            coefs_new
                Coefficients from the current iteration.
            tol : float, optional
                Tolerance level for convergence.

        Returns
        -------
            True if the coefficients have converged (for the given tolerance), False otherwise.

        """
        if self.iter == 0:
            return False
        return np.linalg.norm(coefs_new - coefs_old, ord=2).item() < tol

    def get_node_results(self) -> list[tuple[np.ndarray, np.ndarray]]:
        """
        Get the node results from the object.

        Returns
        -------
            The node results stored in the object.

        """
        return self.node_results

    def get_aggregated_results(self) -> dict[str, np.ndarray]:
        """
        Get the aggregated results from the object.

        Returns
        -------
            A dict with results.
            The coefficients are stored under the key "coef".
            If standard errors, z-scores and p-values have been calculated,
                they are stored under the keys "se", "z" and "p", respectively.


        Raises
        ------
            ValueError: If no aggregated results are available.
        """
        if self.iter == 0:
            raise ValueError("No aggregated results available. Please run aggregate_results first.")

        res = {
            "coef": self.coefs,
        }

        if self.info_calculated:
            res["se"] = self.se_coefs
            res["z"] = self.z_scores
            res["p"] = self.p_values

        return res

    def get_coefs(self) -> np.ndarray:
        """
        Retrieve the aggregated coefficients from the object.

        Returns
        -------
            The aggregated coefficients stored in the object.
        """
        return self.coefs

    def calc_info(self) -> None:
        """Calculate standard errors, z-scores and p-values for the aggregated coefficients and store them."""
        self.info_calculated = True
        covariance_matrix = np.linalg.inv(self.fisher_info_agg)
        self.se_coefs = np.sqrt(np.diag(covariance_matrix))

        # Compute Wald z-scores and p-values:
        self.z_scores = self.coefs / self.se_coefs
        self.p_values = 2 * (1 - norm.cdf(np.abs(self.z_scores)))

    def get_summary(self) -> dict:
        """
        Get a summary of the aggregated results, including coefficients, standard errors, z-scores and p-values.

        This method assumes that aggregate_results has been called.
        It calculates standard errors, z-scores and p-values from the aggregated results.

        Returns
        -------
            A dict with summary results.
            The coefficients are stored under the key "coef".
            Standard errors, z-scores and p-values are stored under the keys "se", "z" and "p", respectively.
        """
        self.calc_info()
        return dict(coef=self.coefs, se=self.se_coefs, z=self.z_scores, p=self.p_values)
