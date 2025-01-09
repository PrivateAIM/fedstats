import pytest
import numpy as np
from fedstats.aggregation import MetaAnalysisAggregator, MetaAnalysisAggregatorUnit, MetaAnalysisAggregatorCollection

def test_aggregator_unit():
    """
    Test MetaAnalysisAggregatorUnit for a single meta-analysis.
    """
    results = [(1., 0.1),  # Server 1 
               (2., 0.1),  # Server 2 
               (3., 0.1),  # Server 3 
               (4., 0.1)   # Server 4 
               ] 
    meta_analysis = MetaAnalysisAggregatorUnit(results)
    meta_analysis.aggregate_results(calculate_heterogeneity=True)
    agg_res = meta_analysis.get_results()
    
    var_agg = 1 / (1 / 0.1 * 4)  # Variance of pooled effect size
    assert np.allclose(2.5, agg_res["aggregated_results"])  # Check pooled effect size
    assert np.allclose(var_agg, agg_res["aggregated_variance"])  # Check pooled variance
    assert agg_res["confidence_interval"][0] < 2.5 < agg_res["confidence_interval"][1]  # CI includes pooled effect size



def test_aggregator_collection():
    """
    Test MetaAnalysisAggregatorCollection for multiple meta-analyses.
    """
    results = [
        [(1., 0.1), (1., 0.1)],  # Server 1
        [(2., 0.1), (2., 0.1)],  # Server 2
        [(3., 0.1), (3., 0.1)],  # Server 3
        [(4., 0.1), (4., 0.1)],  # Server 4
    ]

    collection = MetaAnalysisAggregatorCollection(results)
    collection.aggregate_results(calculate_heterogeneity=True)
    agg_res = collection.get_results()

    # Expected pooled results
    expected_effects = [2.5] * 2
    expected_variances = [1 / (1 / 0.1 * 4)] * 2  # Same for both effect sizes

    assert np.allclose(agg_res["aggregated_results"], expected_effects)
    assert np.allclose(agg_res["aggregated_variance"], expected_variances)
    for i, ci in enumerate(agg_res["confidence_interval"]):
        assert ci[0] < expected_effects[i] < ci[1]



def test_meta_analysis_aggregator():
    """
    Test MetaAnalysisAggregator for wrapper behavior over Unit and Collection.
    """
    # Test as Unit
    unit_results = [(1., 0.1), (2., 0.1), (3., 0.1), (4., 0.1)]
    aggregator_unit = MetaAnalysisAggregator(unit_results)
    assert aggregator_unit.isunit is True  # Verify it's treated as a unit
    aggregator_unit.aggregate_results(calculate_heterogeneity=True)
    agg_res_unit = aggregator_unit.get_results()
    
    var_agg_unit = 1 / (1 / 0.1 * 4)
    assert np.allclose(2.5, agg_res_unit["aggregated_results"])
    assert np.allclose(var_agg_unit, agg_res_unit["aggregated_variance"])
    assert agg_res_unit["confidence_interval"][0] < 2.5 < agg_res_unit["confidence_interval"][1]

    # Test as Collection
    collection_results = [
        [(1., 0.1), (1., 0.1)],  # Server 1
        [(2., 0.1), (2., 0.1)],  # Server 2
        [(3., 0.1), (3., 0.1)],  # Server 3
        [(4., 0.1), (4., 0.1)],  # Server 4
    ]
    aggregator_collection = MetaAnalysisAggregator(collection_results)
    assert aggregator_collection.isunit is False  # Verify it's treated as a collection
    aggregator_collection.aggregate_results(calculate_heterogeneity=True)
    agg_res_collection = aggregator_collection.get_results()

    expected_effects = [2.5] * 2
    expected_variances = [1 / (1 / 0.1 * 4)] * 2  # Same for both effect sizes

    assert np.allclose(agg_res_collection["aggregated_results"], expected_effects)
    assert np.allclose(agg_res_collection["aggregated_variance"], expected_variances)
    for i, ci in enumerate(agg_res_collection["confidence_interval"]):
        assert ci[0] < expected_effects[i] < ci[1]


    # test wrong input format
    unit_results = [1,2,3]
    with pytest.raises(TypeError):
        MetaAnalysisAggregator(unit_results)
