from unittest.mock import patch

import numpy as np
import pytest


def test_split_data_num_clients_and_labels():
    """
    Test the split_data function to ensure it correctly splits features and labels.
    """
    from fedstats.util import split_data

    # Create dummy data
    X = np.array([[1, 2], [2, 4], [3, 6], [4, 8]])
    y = np.array([1, 2, 3, 4])

    for num_clients in [1, 2, 4]:
        X_split, y_split = split_data(X, y, num_clients=num_clients, random_state=42)

        # Check that the number of splits matches num_clients
        assert len(X_split) == num_clients
        assert len(y_split) == num_clients

        # Check that the total number of samples is preserved
        total_samples = sum(len(x) for x in X_split)
        assert total_samples == len(X)

        for x_part, y_part in zip(X_split, y_split, strict=False):
            assert len(x_part) == len(y_part)

            for i in range(len(x_part)):
                assert y_part[i] == x_part[i][0]  # since y is the first column of X in the dummy data


def test_simulate_logistic_regression_output_shapes():
    """
    Test the simulate_logistic_regression function to ensure it returns correct shapes.
    """
    from fedstats.util import simulate_logistic_regression

    X_splits, y_splits = simulate_logistic_regression(random_state=42, n=150, p=5)

    total_samples = sum(len(x) for x in X_splits)
    assert total_samples == 150

    for x_part, y_part in zip(X_splits, y_splits, strict=False):
        assert len(x_part) == len(y_part)
        assert x_part.shape[1] == 5  # number of features


def test_simulate_logistic_regression_labels_binary():
    """
    Test the simulate_logistic_regression function to ensure labels are binary.
    """
    from fedstats.util import simulate_logistic_regression

    _, y_splits = simulate_logistic_regression(random_state=42, n=200, p=4)

    for y_part in y_splits:
        for label in y_part:
            assert label in [0, 1]


def test_simulate_poisson_regression_output_shapes():
    """
    Test the simulate_poisson_regression function to ensure it returns correct shapes.
    """
    from fedstats.util import simulate_poisson_regression

    X_splits, y_splits = simulate_poisson_regression(n=120, p=3)

    total_samples = sum(len(x) for x in X_splits)
    assert total_samples == 120

    for x_part, y_part in zip(X_splits, y_splits, strict=False):
        assert len(x_part) == len(y_part)
        assert x_part.shape[1] == 3  # number of features


def test_simulate_gaussian_regression_output_shapes():
    """
    Test the simulate_gaussian_regression function to ensure it returns correct shapes.
    """
    from fedstats.util import simulate_gaussian_regression

    X_splits, y_splits = simulate_gaussian_regression(n=180, p=6)

    total_samples = sum(len(x) for x in X_splits)
    assert total_samples == 180

    for x_part, y_part in zip(X_splits, y_splits, strict=False):
        assert len(x_part) == len(y_part)
        assert x_part.shape[1] == 6  # number of features


def test_plot_forest_invalid_input():
    """
    Test the plot_forest function to ensure it raises ValueError for invalid input lengths.
    """
    from fedstats.util import plot_forest

    # Mismatched lengths
    data = [
        (np.array([1, 2]), np.array([0.5]), np.array([1.5, 2.5])),  # lower_bounds length mismatch
    ]

    with patch("matplotlib.pyplot.show"), pytest.raises(ValueError):
        plot_forest(data)


def test_plot_forest_valid_input():
    """
    Test the plot_forest function with valid input to ensure no exceptions are raised.
    """
    from fedstats.util import plot_forest

    data = [
        (np.array([1, 2]), np.array([0.5, 1.5]), np.array([1.5, 2.5])),
    ]

    with patch("matplotlib.pyplot.show"):
        plot_forest(data)
