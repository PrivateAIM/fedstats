import pytest
import numpy as np
from scipy.special import expit
from fedstats.models.local_fisher_scoring import LocalFisherScoring


def test_local_fisher_scoring_initialization():
    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([0, 1, 0])
    model = LocalFisherScoring(X, y, family="binomial")
    assert model.X.shape == (3, 3)  # Intercept added
    assert model.y.shape == (3,)
    assert model.beta.shape == (3,)
    assert model.family == "binomial"
    assert not model.finished


def test_make_covariate_matrix():
    X = np.array([[1, 2], [3, 4]])
    model = LocalFisherScoring(X, np.array([1, 0]), family="binomial")
    new_X = model.make_covariate_matrix(X, fit_intercept=True)
    assert new_X.shape == (2, 3)
    assert np.all(new_X[:, 0] == 1)  # Check intercept column


def test_make_covariate_matrix_standardize():
    X = np.array([[1, 2], [3, 4]])
    model = LocalFisherScoring(X, np.array([1, 0]), family="binomial")
    with pytest.raises(NotImplementedError):
        model.make_covariate_matrix(X, fit_intercept=True, standardize=True)


def test_get_glm_functions():
    model = LocalFisherScoring(np.array([[1, 2]]), np.array([0]), family="binomial")
    mu_fn, dmu_deta_fn, V_fn = model.get_glm_functions("binomial")
    eta = np.array([0.5, -0.5])
    assert np.allclose(mu_fn(eta), expit(eta))
    assert np.allclose(dmu_deta_fn(eta), expit(eta) * (1 - expit(eta)))
    assert np.allclose(V_fn(expit(eta)), expit(eta) * (1 - expit(eta)))


def test_get_glm_functions_gaussian():
    model = LocalFisherScoring(np.array([[1, 2]]), np.array([0]), family="gaussian")
    mu_fn, dmu_deta_fn, V_fn = model.get_glm_functions("gaussian")
    eta = np.array([0.5, -0.5])
    assert np.allclose(mu_fn(eta), eta)
    assert np.allclose(dmu_deta_fn(eta), np.ones_like(eta))
    assert np.allclose(V_fn(eta), np.ones_like(eta))


def test_get_glm_functions_poisson():
    model = LocalFisherScoring(np.array([[1, 2]]), np.array([0]), family="poisson")
    mu_fn, dmu_deta_fn, V_fn = model.get_glm_functions("poisson")
    eta = np.array([0.5, -0.5])
    assert np.allclose(mu_fn(eta), np.exp(eta))
    assert np.allclose(dmu_deta_fn(eta), np.exp(eta))
    assert np.allclose(V_fn(np.exp(eta)), np.exp(eta))


def test_calc_fisher_scoring_parts():
    X = np.array([[1, 2], [3, 4]])
    y = np.array([1, 0])
    model = LocalFisherScoring(X, y, family="binomial")
    model.set_coefs(np.array([0.1, 0.2, 0.3]))
    Fisher_info, rhs = model.calc_fisher_scoring_parts()
    assert Fisher_info.shape == (3, 3)
    assert rhs.shape == (3,)


def test_get_glm_functions_invalid_family():
    with pytest.raises(
        ValueError,
        match="Unsupported family. Choose from 'gaussian', 'binomial', 'poisson'.",
    ):
        LocalFisherScoring(np.array([[1, 2]]), np.array([0]), family="invalid_family")
