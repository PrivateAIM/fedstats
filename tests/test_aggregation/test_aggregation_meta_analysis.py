import numpy as np
import pytest

from fedstats.aggregation.meta_analysis import (
    MetaAnalysisAggregator,
    MetaAnalysisAggregatorCollection,
    MetaAnalysisAggregatorUnit,
)


def test_aggregator_unit():
    """
    Test MetaAnalysisAggregatorUnit for a single meta-analysis.
    """
    results = [
        (1.0, 0.1),  # Server 1
        (2.0, 0.1),  # Server 2
        (3.0, 0.1),  # Server 3
        (4.0, 0.1),  # Server 4
    ]
    meta_analysis = MetaAnalysisAggregatorUnit(results)
    meta_analysis.aggregate_results(calculate_heterogeneity=True)
    agg_res = meta_analysis.get_aggregated_results()

    # Variance of pooled effect size
    var_agg = 1 / (1 / 0.1 * 4)

    # Check pooled effect size
    assert np.allclose(2.5, agg_res["aggregated_results"])

    # Check pooled variance
    assert np.allclose(var_agg, agg_res["aggregated_variance"])

    # CI includes pooled effect size
    assert agg_res["confidence_interval"][0] < 2.5 < agg_res["confidence_interval"][1]  # type: ignore


def test_aggregator_unit_no_heterogeneity():
    """
    Test MetaAnalysisAggregatorUnit without calculating heterogeneity.
    """
    results = [
        (1.0, 0.1),  # Server 1
        (2.0, 0.1),  # Server 2
        (3.0, 0.1),  # Server 3
        (4.0, 0.1),  # Server 4
    ]
    meta_analysis = MetaAnalysisAggregatorUnit(results)
    meta_analysis.aggregate_results(calculate_heterogeneity=False)
    agg_res = meta_analysis.get_aggregated_results()

    var_agg = 1 / (1 / 0.1 * 4)
    assert np.allclose(2.5, agg_res["aggregated_results"])
    assert np.allclose(var_agg, agg_res["aggregated_variance"])
    assert agg_res["confidence_interval"][0] < 2.5 < agg_res["confidence_interval"][1]  # type: ignore

    # q_statistic should not be calculated
    assert "q_statistic" not in agg_res


def test_aggregator_collection():
    """
    Test MetaAnalysisAggregatorCollection for multiple meta-analyses.
    """
    results = [
        [(1.0, 0.1), (1.0, 0.1)],  # Server 1
        [(2.0, 0.1), (2.0, 0.1)],  # Server 2
        [(3.0, 0.1), (3.0, 0.1)],  # Server 3
        [(4.0, 0.1), (4.0, 0.1)],  # Server 4
    ]

    collection = MetaAnalysisAggregatorCollection(results)
    collection.aggregate_results(calculate_heterogeneity=True)
    agg_res = collection.get_aggregated_results()

    # Expected pooled results
    expected_effects = [2.5] * 2
    expected_variances = [1 / (1 / 0.1 * 4)] * 2  # Same for both effect sizes

    assert np.allclose(agg_res["aggregated_results"], expected_effects)
    assert np.allclose(agg_res["aggregated_variance"], expected_variances)
    for i, ci in enumerate(agg_res["confidence_interval"]):
        assert ci[0] < expected_effects[i] < ci[1]  # type: ignore

    # Check that q_statistic is calculated
    assert "q_statistic" in agg_res


def test_aggregator_collection_no_heterogeneity():
    """
    Test MetaAnalysisAggregatorCollection without calculating heterogeneity.
    """
    results = [
        [(1.0, 0.1), (1.0, 0.1)],  # Server 1
        [(2.0, 0.1), (2.0, 0.1)],  # Server 2
        [(3.0, 0.1), (3.0, 0.1)],  # Server 3
        [(4.0, 0.1), (4.0, 0.1)],  # Server 4
    ]

    collection = MetaAnalysisAggregatorCollection(results)
    collection.aggregate_results(calculate_heterogeneity=False)
    agg_res = collection.get_aggregated_results()

    expected_effects = [2.5] * 2
    expected_variances = [1 / (1 / 0.1 * 4)] * 2

    assert np.allclose(agg_res["aggregated_results"], expected_effects)
    assert np.allclose(agg_res["aggregated_variance"], expected_variances)
    for i, ci in enumerate(agg_res["confidence_interval"]):
        assert ci[0] < expected_effects[i] < ci[1]  # type: ignore

    # q_statistic should not be calculated
    assert "q_statistic" not in agg_res


def test_meta_analysis_aggregator_unit_wrapper():
    """
    Test MetaAnalysisAggregator for wrapper behavior over Unit.
    """
    unit_results = [(1.0, 0.1), (2.0, 0.1), (3.0, 0.1), (4.0, 0.1)]
    aggregator_unit = MetaAnalysisAggregator(unit_results)
    assert aggregator_unit.isunit is True  # Verify it's treated as a unit
    aggregator_unit.aggregate_results(calculate_heterogeneity=True)
    agg_res_unit = aggregator_unit.get_aggregated_results()

    var_agg_unit = 1 / (1 / 0.1 * 4)
    assert np.allclose(2.5, agg_res_unit["aggregated_results"])
    assert np.allclose(var_agg_unit, agg_res_unit["aggregated_variance"])
    assert agg_res_unit["confidence_interval"][0] < 2.5 < agg_res_unit["confidence_interval"][1]  # type: ignore


def test_meta_analysis_aggregator_collection_wrapper():
    """
    Test MetaAnalysisAggregator for wrapper behavior over Collection.
    """
    collection_results = [
        [(1.0, 0.1), (1.0, 0.1)],  # Server 1
        [(2.0, 0.1), (2.0, 0.1)],  # Server 2
        [(3.0, 0.1), (3.0, 0.1)],  # Server 3
        [(4.0, 0.1), (4.0, 0.1)],  # Server 4
    ]
    aggregator_collection = MetaAnalysisAggregator(collection_results)
    assert aggregator_collection.isunit is False  # Verify it's treated as a collection
    aggregator_collection.aggregate_results(calculate_heterogeneity=True)
    agg_res_collection = aggregator_collection.get_aggregated_results()

    expected_effects = [2.5] * 2
    expected_variances = [1 / (1 / 0.1 * 4)] * 2  # Same for both effect sizes

    assert np.allclose(agg_res_collection["aggregated_results"], expected_effects)
    assert np.allclose(agg_res_collection["aggregated_variance"], expected_variances)
    for i, ci in enumerate(agg_res_collection["confidence_interval"]):  # type: ignore
        assert ci[0] < expected_effects[i] < ci[1]


def test_meta_analysis_aggregator_rejects_wrong_input():
    """
    Test MetaAnalysisAggregator rejects wrong input format.
    """
    unit_results = [1, 2, 3]
    with pytest.raises(TypeError):
        MetaAnalysisAggregator(unit_results)
