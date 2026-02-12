import pytest
import numpy as np
from scipy.stats import norm
from fedstats.aggregation.fed_glm import FedGLM


def test_fedglm_initialization():
    model = FedGLM()
    assert model.node_results is None
    assert np.isinf(model.coefs).all()
    assert model.iter == 0


def test_set_results():
    results = [(np.array([[1, 2], [3, 4]]), np.array([1, 0]))]
    model = FedGLM()
    model.set_node_results(results)
    assert model.node_results == results


def test_aggregate_results():
    results = [
        (np.array([[2.0, 0.0], [0.0, 2.0]]), np.array([2.0, 4.0])),
        (np.array([[1.0, 1.0], [1.0, 3.0]]), np.array([1.0, 3.0])),
    ]
    model = FedGLM(results)
    model.aggregate_results()
    expected_coefs = np.linalg.solve(model.fisher_info_agg, model.rhs_agg)
    assert np.allclose(model.coefs, expected_coefs)
    assert model.iter == 1


def test_aggregate_results_singular_matrix():
    results = [
        (np.array([[1, 1], [1, 1]]), np.array([1, 1]))  # Singular matrix
    ]
    model = FedGLM(results)
    model.aggregate_results()
    assert model.coefs.shape == (2,)  # Should still return coefficients


def test_check_convergence():
    model = FedGLM([])
    coefs_old = np.array([1.0, 2.0])
    coefs_new = np.array([1.0000001, 2.0000001])
    model.iter = 1  # Ensure it's not the first iteration
    assert model.check_convergence(coefs_old, coefs_new, tol=1e-6) is True
    assert model.check_convergence(coefs_old, coefs_new, tol=1e-9) is False


def test_check_convergence_first_iter():
    model = FedGLM([])
    coefs_old = np.array([1.0, 2.0])
    coefs_new = np.array([1.1, 2.1])
    assert model.check_convergence(coefs_old, coefs_new) is False


def test_get_results():
    results = [(np.array([[1, 2], [3, 4]]), np.array([1, 0]))]
    model = FedGLM(results)
    assert model.get_node_results() == results


def test_get_coefs():
    results = [(np.array([[2.0, 0.0], [0.0, 2.0]]), np.array([2.0, 4.0]))]
    model = FedGLM(results)
    model.aggregate_results()
    assert np.allclose(model.get_coefs(), model.coefs)


def test_calc_info():
    results = [(np.array([[2.0, 0.0], [0.0, 2.0]]), np.array([2.0, 4.0]))]
    model = FedGLM(results)
    model.aggregate_results(calc_info=True)
    assert hasattr(model, "se_coefs")
    assert hasattr(model, "z_scores")
    assert hasattr(model, "p_values")
    assert model.se_coefs.shape == model.coefs.shape
    assert model.z_scores.shape == model.coefs.shape
    assert model.p_values.shape == model.coefs.shape


def test_get_summary():
    results = [(np.array([[2.0, 0.0], [0.0, 2.0]]), np.array([2.0, 4.0]))]
    model = FedGLM(results)
    model.aggregate_results()
    summary = model.get_summary()
    assert "coef" in summary
    assert "se" in summary
    assert "z" in summary
    assert "p" in summary

def test_get_aggregated_results_before_aggregation():
    model = FedGLM([])
    with pytest.raises(ValueError):
        model.get_aggregated_results()

def test_get_aggregated_results_after_aggregation():
    results = [(np.array([[2.0, 0.0], [0.0, 2.0]]), np.array([2.0, 4.0]))]
    model = FedGLM(results)
    model.aggregate_results(calc_info=True)
    agg_results = model.get_aggregated_results()
    assert "coef" in agg_results
    assert np.allclose(agg_results["coef"], model.coefs)
    assert "se" in agg_results
    assert "z" in agg_results
    assert "p" in agg_results