
from abc import ABC, abstractmethod
from typing import List, Tuple, Union, Dict
import numpy as np

# Abstract class for aggregation methods
class Aggregator(ABC):
    def __init__(self, results: Union[List[Tuple], List[List[Tuple]]]) -> None:
        """
        Initialize the aggregator with results.

        :param results: A list of tuples or a list of lists of tuples.
        """
        self.results = results

    @abstractmethod
    def aggregate_results(self, *args, **kwargs) -> None:  # pragma: no cover
        """
        Abstract method to that defines how results are aggregated.
        """
        pass

    @abstractmethod
    def get_results(self, *args, **kwargs) -> Union[Dict[str, np.ndarray], Dict[str, Union[float, Tuple[float, float]]]]:  # pragma: no cover
        """
        Abstract method to return aggregated results.

        :return: Aggregated results.
        """
        pass

