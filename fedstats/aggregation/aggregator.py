from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Union

import numpy as np


# Abstract class for aggregation methods
class Aggregator(ABC):
    def __init__(self, node_results: Union[List[Tuple], List[List[Tuple]], None]) -> None:
        """
        Initialize the aggregator with results.

        :param results: A list of tuples or a list of lists of tuples.
        """
        self.node_results = node_results

    @abstractmethod
    def aggregate_results(self, *args, **kwargs) -> None:  # pragma: no cover
        """
        Abstract method to that defines how results are aggregated.
        """
        pass

    @abstractmethod
    def get_aggregated_results(
        self, *args, **kwargs
    ) -> Union[
        Dict[str, np.ndarray],
        Dict[str, Union[float, Tuple[float, float]]] | tuple | list | float,
    ]:  # pragma: no cover
        """
        Abstract method to return aggregated results.

        :return: Aggregated results.
        """
        pass
