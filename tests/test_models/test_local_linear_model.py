import pytest
import numpy as np
from fedstats.models.local_linear_regression import LocalLinearRegression


def test_make_covariate_matrix():
    X = np.array([[1, 2], [3, 4]])
    model = LocalLinearRegression(X, np.array([1, 2]))
    assert np.array_equal(
        model.make_covariate_matrix(X, True), np.array([[1, 1, 2], [1, 3, 4]])
    )
    assert np.array_equal(model.make_covariate_matrix(X, False), X)

    with pytest.raises(NotImplementedError):
        model.make_covariate_matrix(X, True, standardize=True)


def test_fit():
    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([1, 2, 3])
    model = LocalLinearRegression(X, y)
    model.fit()
    assert model.model_fitted is True


def test_get_result():
    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([1, 2, 3])
    model = LocalLinearRegression(X, y)

    with pytest.raises(Exception, match="Fit model first."):
        model.get_result()

    model.fit()
    result = model.get_result()
    assert isinstance(result, list)
    assert all(isinstance(t, tuple) and len(t) == 2 for t in result)


def test_get_result_exceptions():
    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([1, 2, 3])
    model = LocalLinearRegression(X, y)
    model.fit()

    with pytest.raises(Exception, match="Either use_sd or use_n has to be True."):
        model.get_result(use_sd=False, use_n=False)


def test_fit_intercept():
    X = np.array([[1], [2], [3]])
    y = np.array([1, 2, 3])
    model = LocalLinearRegression(X, y, fit_intercept=False)
    assert model.X.shape == X.shape  # No intercept added
    model_with_intercept = LocalLinearRegression(X, y, fit_intercept=True)
    assert model_with_intercept.X.shape[1] == X.shape[1] + 1  # Intercept added


def test_get_result_with_n():
    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([1, 2, 3])
    model = LocalLinearRegression(X, y)
    model.fit()
    result = model.get_result(use_sd=False, use_n=True)
    assert isinstance(result, list)
    assert all(isinstance(t, tuple) and len(t) == 2 for t in result)
    assert all(t[1] == model.n for t in result)
