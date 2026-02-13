import numpy as np
from scipy.stats import norm
from functools import reduce
from typing import Dict, Union
from fedstats.aggregation.aggregator import Aggregator


class FedGLM(Aggregator):
    info_calculated: bool = False

    def __init__(self, node_results: Union[list, None] = None) -> None:
        """
        Handels aggregation for of GLM fisher scorings

        Idea is from: https://www.mdpi.com/1999-4893/15/7/243

        :param results: A list of np.arrays that represent local results from calcualte_fisher_scoring_parts
        """

        super().__init__(node_results)
        self.coefs = np.array(np.inf)  # TODO: Make a better solution for init
        self.iter = 0

    def set_node_results(self, node_results: list[tuple[np.ndarray, np.ndarray]]) -> None:
        self.node_results = node_results

    def aggregate_results(self, calc_info: bool = False) -> None:
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
        if self.iter == 0:
            return False
        return np.linalg.norm(coefs_new - coefs_old, ord=2).item() < tol

    def get_node_results(self) -> list[tuple[np.ndarray, np.ndarray]]:
        return self.node_results

    def get_aggregated_results(self) -> Dict[str, np.ndarray]:
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
        return self.coefs

    def calc_info(self) -> None:
        self.info_calculated = True
        covariance_matrix = np.linalg.inv(self.fisher_info_agg)
        self.se_coefs = np.sqrt(np.diag(covariance_matrix))

        # Compute Wald z-scores and p-values:
        self.z_scores = self.coefs / self.se_coefs
        self.p_values = 2 * (1 - norm.cdf(np.abs(self.z_scores)))

    def get_summary(self) -> dict:
        self.calc_info()
        return dict(coef=self.coefs, se=self.se_coefs, z=self.z_scores, p=self.p_values)
