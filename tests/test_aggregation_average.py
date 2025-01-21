import pytest
import numpy as np
from fedstats.aggregation.average import AverageAggregator, AverageAggregatorUnit, AverageAggregatorCollection

def test_aggregator_unit():
    """
    Test AverageAggregatorUnit for a single meta-analysis.
    """
    results = [(1., 10),  # Server 1 
               (2., 20),  # Server 2 
               (3., 30),  # Server 3 
               (4., 40)   # Server 4 
               ] 
    meta_analysis = AverageAggregatorUnit(results)
    meta_analysis.aggregate_results()
    agg_res = meta_analysis.get_results()
    
    assert np.allclose(3., agg_res["aggregated_results"])  # Check pooled effect size



def test_aggregator_collection():
    """
    Test AverageAggregatorCollection for multiple meta-analyses.
    """
    results = [
        [(1., 10), (1., 10)],  # Server 1
        [(2., 20), (2., 20)],  # Server 2
        [(3., 30), (3., 30)],  # Server 3
        [(4., 40), (4., 40)],  # Server 4
    ]

    collection = AverageAggregatorCollection(results)
    collection.aggregate_results()
    agg_res = collection.get_results()

    expected_effects = [3.] * 2  # Expected pooled results
    assert np.allclose(agg_res["aggregated_results"], expected_effects)



def test_average_aggregator():
    """
    Test AverageAggregator for wrapper behavior over Unit and Collection.
    """
    # Test as Unit
    unit_results = [(1., 10), (2., 20), (3., 30), (4., 40)]
    aggregator_unit = AverageAggregator(unit_results)
    assert aggregator_unit.isunit is True  # Verify it's treated as a unit

    aggregator_unit.aggregate_results()
    agg_res_unit = aggregator_unit.get_results()
    assert np.allclose(3., agg_res_unit["aggregated_results"])

    # Test as Collection
    collection_results = [
        [(1., 10), (1., 10)],  # Server 1
        [(2., 20), (2., 20)],  # Server 2
        [(3., 30), (3., 30)],  # Server 3
        [(4., 40), (4., 40)],  # Server 4
    ]
    aggregator_collection = AverageAggregator(collection_results)
    assert aggregator_collection.isunit is False  # Verify it's treated as a collection

    aggregator_collection.aggregate_results()
    agg_res_collection = aggregator_collection.get_results()
    expected_effects = [3.] * 2
    assert np.allclose(agg_res_collection["aggregated_results"], expected_effects)


    # test wrong input format
    unit_results = [1,2,3]
    with pytest.raises(TypeError):
        AverageAggregator(unit_results)
