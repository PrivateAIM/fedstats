from fedstats.aggregation.aggregator import Aggregator


def test_aggregator_unit():
    """
    Test abstract class Aggregator
    """
    # arbitrary results from servers
    node_results = [
        (-99, -99),  # Server 1
        (-99, -99),  # Server 2
    ]
    Aggregator.__abstractmethods__ = set()  # type: ignore

    agg = Aggregator(node_results)  # type: ignore
    agg.aggregate_results()
    agg_results = agg.get_aggregated_results()

    assert isinstance(agg, Aggregator)
    assert isinstance(agg.node_results, list)
    assert agg.node_results == node_results
    assert agg_results is None
