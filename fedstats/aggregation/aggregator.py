"""Abstract base class for aggregation methods in federated statistics."""

from abc import ABC, abstractmethod
from typing import Any


class Aggregator(ABC):
    """Abstract base class for aggregation methods."""

    def __init__(self, node_results: list[tuple] | list[list[tuple]] | None) -> None:
        """Initialize the aggregator with results.

        Parameters
        ----------
            node_results
                Local results from the nodes. The format depends on the specific aggregator implementation.
        """
        self.node_results = node_results

    @abstractmethod
    def aggregate_results(self, *args, **kwargs) -> None:  # pragma: no cover
        """Abstract method to perform aggregation."""
        pass

    @abstractmethod
    def get_aggregated_results(self, *args, **kwargs) -> Any:  # pragma: no cover
        """Abstract method to return aggregated results.

        Returns
        -------
            Aggregated results in a format depending on the specific aggregator implementation.
        """
        pass
