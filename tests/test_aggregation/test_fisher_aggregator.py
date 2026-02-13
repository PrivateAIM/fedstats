import pytest

from fedstats.aggregation.fisher_aggregator import FisherAggregator


def test_initialization():
    results = [(0.5, 0.1), (0.3, 0.2)]
    aggregator = FisherAggregator(results)
    assert aggregator.node_results == results
    assert not hasattr(aggregator, "combined_p_value")


def test_initialization_no_results():
    aggregator = FisherAggregator(None)  # type: ignore
    assert aggregator.node_results is None
    assert not hasattr(aggregator, "combined_p_value")


def test_aggregation_no_results():
    aggregator = FisherAggregator(None)  # type: ignore
    with pytest.raises(ValueError) as exc_info:
        aggregator.aggregate_results()
    assert str(exc_info.value) == "No results to aggregate."


def test_aggregation_empty_results():
    aggregator = FisherAggregator([])
    with pytest.raises(ValueError) as exc_info:
        aggregator.aggregate_results()
    assert str(exc_info.value) == "No results to aggregate."


def test_aggregation_and_retrieval():
    results = [(0.5, 0.1), (0.3, 0.2)]
    aggregator = FisherAggregator(results)
    aggregator.aggregate_results()
    combined_p = aggregator.get_aggregated_results()
    assert isinstance(combined_p, float)
    assert 0 <= combined_p <= 1


def test_get_aggregated_results_before_aggregation():
    results = [(0.5, 0.1), (0.3, 0.2)]
    aggregator = FisherAggregator(results)
    with pytest.raises(ValueError) as exc_info:
        aggregator.get_aggregated_results()
    assert str(exc_info.value) == "No aggregated result computed. Call aggregate_results first."


def test_aggregation_verbose(capsys):
    results = [(0.5, 0.1), (0.3, 0.2)]
    aggregator = FisherAggregator(results)
    aggregator.aggregate_results(verbose=True)
    captured = capsys.readouterr()
    assert "Site 0 local p-value:" in captured.out
    assert "Site 1 local p-value:" in captured.out
    combined_p = aggregator.get_aggregated_results()
    assert isinstance(combined_p, float)
    assert 0 <= combined_p <= 1


def test_aggregation_one_zero_stddev():
    import numpy as np

    results = [(0.5, 0.0), (0.3, 0.2)]
    aggregator = FisherAggregator(results)
    with pytest.warns(UserWarning, match="Standard deviation of zero encountered"):
        aggregator.aggregate_results()

    combined_p = aggregator.get_aggregated_results()
    assert isinstance(combined_p, float)
    assert np.isnan(combined_p)


def test_estimate_to_pvalue():
    import numpy as np
    from scipy.stats import norm

    est = 0.5
    sd = 0.1
    expected_p = 2 * (1 - norm.cdf(abs(est / sd)))
    p_val = FisherAggregator._estimate_to_pvalue(est, sd)
    np.testing.assert_almost_equal(p_val, expected_p)


def test_estimate_to_pvalue_array():
    import numpy as np
    from scipy.stats import norm

    est = np.array([0.5, 0.5])
    sd = np.array([0.1, 0.7])
    expected_p = 2 * (1 - norm.cdf(np.abs(est / sd)))
    p_val = FisherAggregator._estimate_to_pvalue(est, sd)
    np.testing.assert_almost_equal(p_val, expected_p)


def test_estimate_to_pvalue_zero_sd():
    import numpy as np

    est = 0.5
    sd = 0.0
    with pytest.warns(UserWarning, match="Standard deviation of zero encountered"):
        p_val = FisherAggregator._estimate_to_pvalue(est, sd)
    assert np.isnan(p_val)
