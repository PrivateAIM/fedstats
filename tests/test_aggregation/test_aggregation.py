import pytest
import numpy as np
from fedstats.aggregation.aggregator import Aggregator

def test_aggregator_unit():
    """
    Test abstract class Aggregator
    """
    # arbitrary results from servers
    results = [(-99, -99),  # Server 1 
               (-99, -99),  # Server 2 
               ]
    Aggregator.__abstractmethods__ = set()

    agg = Aggregator(results)
    # agg.aggregate_results() 
    # agg.get_results() 
    
    assert isinstance(agg, Aggregator)
    assert isinstance(agg.node_results, list)

    
