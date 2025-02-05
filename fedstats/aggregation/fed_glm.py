import numpy as np
from scipy.stats import norm
from functools import reduce
from fedstats.aggregation.aggregator import Aggregator


class FedGLM(Aggregator):
    def __init__(self, results: list) -> None:
        """
        Handels aggregation for of GLM fisher scorings

        :param results: A list of np.arrays that represent local results from calcualte_fisher_scoring_parts
        """

        super().__init__(results)
        self.coefs = np.array(np.inf)  # TODO: Make a better solution for init
        self.iter = 0

    def set_results(self, results: list[tuple[np.ndarray, np.ndarray]]) -> None:
        self.results = results

    def aggregate_results(self, calc_info: bool = False) -> None:
        fisher_infos = [res[0] for res in self.results]
        rhss = [res[1] for res in self.results]

        fisher_info_agg = reduce(lambda x, y: x + y, fisher_infos)
        rhs_agg = reduce(lambda x, y: x + y, rhss)
        try:
            coefs = np.linalg.solve(fisher_info_agg, rhs_agg)
        except np.linalg.LinAlgError:
            coefs = np.linalg.pinv(fisher_info_agg) @ rhs_agg
        self.coefs = coefs
        self.iter += 1

        if calc_info:
            self.calc_info(fisher_info_agg, coefs)

    def check_convergence(
        self, coefs_old: np.ndarray, coefs_new: np.ndarray, tol: float = 1e-6
    ) -> bool:
        if self.iter == 0:
            return False
        return np.linalg.norm(coefs_new - coefs_old, ord=2).item() < tol

    def get_results(self) -> list[tuple[np.ndarray, np.ndarray]]:
        return self.results

    def get_coefs(self) -> np.ndarray:
        return self.coefs

    def calc_info(self, Fisher_info: np.ndarray, beta: np.ndarray):
        covariance_matrix = np.linalg.inv(Fisher_info)
        se_beta = np.sqrt(np.diag(covariance_matrix))

        # Compute Wald z-scores and p-values:
        self.z_scores = beta / se_beta
        self.p_values = 2 * (1 - norm.cdf(np.abs(self.z_scores)))
