"""Module for local Fisher scoring computations."""

from collections.abc import Callable

import numpy as np
from scipy.special import expit


class LocalFisherScoring:
    """Class to perform local Fisher scoring computations for GLMs in a federated setting."""

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        family: str,
        fit_intercept: bool = True,
        standardize: bool = False,
    ) -> None:
        """
        Initialize the Local Fisher Scoring calculator.

        Parameters
        ----------
            X : np.ndarray
                The feature matrix (n_samples x n_features).
            y : np.ndarray
                The response vector (n_samples,).
            family : {"gaussian", "binomial", "poisson"}
                The GLM family to use.
            fit_intercept : bool, optional
                Whether to add an intercept term to the model. Default is True.
            standardize : bool, optional
                Whether to standardize the features. Default is False. Not implemented yet.
        """
        self.X = self.make_covariate_matrix(X=X, fit_intercept=fit_intercept, standardize=standardize)
        self.y = y
        self.n, self.p = self.X.shape
        self.beta = np.zeros(self.p)
        self.family = family
        self.iter = 0
        self.finished = False

        self.mu_fn, self.dmu_deta_fn, self.V_fn = self.get_glm_functions(family)

    def make_covariate_matrix(self, X, fit_intercept: bool, standardize: bool = False) -> np.ndarray:
        """
        Prepare the covariate matrix by optionally adding an intercept and standardizing features.

        Parameters
        ----------
            X : np.ndarray
                The original feature matrix.
            fit_intercept : bool
                Whether to add an intercept term.
            standardize : bool
                Whether to standardize the features. Not implemented yet.

        Returns
        -------
            np.ndarray
                The prepared covariate matrix.

        Raises
        ------
            NotImplementedError
                If standardization is requested, as it is not implemented yet.
        """
        if standardize:
            raise NotImplementedError("Not implemented yet. Do it manually beforehand.")
        if fit_intercept:
            X = np.c_[np.ones(X.shape[0]), X]
        return X

    def set_coefs(self, coefs: np.ndarray) -> None:
        """Set the coefficient vector for the model."""
        self.beta = coefs

    def calc_fisher_scoring_parts(self, verbose: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculate the Fisher information matrix and the right-hand side for the current coefficient estimates.

        Parameters
        ----------
            verbose : bool, optional
                Whether to print detailed information during the calculation. Default is False.

        Returns
        -------
            tuple[np.ndarray, np.ndarray]
                A tuple containing the Fisher information matrix and the right-hand side vector.
        """
        if verbose:  # pragma: no cover
            print(f"Calculating iteration {self.iter}...")
        # Get family-specific functions once:
        eta = self.X @ self.beta
        # Evaluate family-specific functions:
        mu = self.mu_fn(eta)
        dmu_deta = self.dmu_deta_fn(eta)
        V_mu = self.V_fn(mu)
        # Safeguard against division by zero:
        dmu_deta = np.where(np.abs(dmu_deta) < 1e-8, 1e-8, dmu_deta)
        V_mu = np.where(np.abs(V_mu) < 1e-8, 1e-8, V_mu)
        # Compute weights: w_i = (dmu/deta)^2 / V(mu)
        weights = (dmu_deta**2) / V_mu
        # Compute the adjusted (working) response:
        z = eta + (self.y - mu) / dmu_deta
        # Construct diagonal weight matrix:
        W = np.diag(weights)
        # Compute Fisher information matrix and the right-hand side:
        XT_W = self.X.T @ W
        Fisher_info = XT_W @ self.X
        rhs = XT_W @ z

        self.iter += 1

        return Fisher_info, rhs

    def get_glm_functions(
        self, family: str
    ) -> tuple[
        Callable[[float | np.ndarray], float | np.ndarray],
        Callable[[float | np.ndarray], float | np.ndarray],
        Callable[[float | np.ndarray], float | np.ndarray],
    ]:
        """Return the functions to compute mu, dmu/deta, and V(mu) for the given GLM family.

        Parameters
        ----------
            family : {"gaussian", "binomial", "poisson"}
                The GLM family to use.

        Returns
        -------
            mu_fn       : function that computes mu given eta.
            dmu_deta_fn : function that computes dmu/deta given eta.
            V_fn        : function that computes V(mu) given mu.

        Raises
        ------
            ValueError: If an unsupported family is provided.
        """

        def gaussian_mu(eta: float | np.ndarray) -> float | np.ndarray:
            """Calculate the identity link: mu = eta."""
            return eta

        def gaussian_dmu_deta(eta: float | np.ndarray) -> float | np.ndarray:
            """Calculate the derivative of identity: dmu/deta = 1."""
            return np.ones_like(eta)

        def gaussian_V(mu: float | np.ndarray) -> float | np.ndarray:
            """Calculate the variance function: constant variance (here set to 1)."""
            return np.ones_like(mu)

        # Define functions for the Binomial (logistic regression) family:
        def binomial_mu(eta: float | np.ndarray) -> float | np.ndarray:
            """Calculate the inverse logit: mu = expit(eta)."""
            return expit(eta)

        def binomial_dmu_deta(eta: float | np.ndarray) -> float | np.ndarray:
            """Calculate the derivative of expit: mu * (1 - mu)."""
            mu = expit(eta)
            return mu * (1 - mu)

        def binomial_V(mu: float | np.ndarray) -> float | np.ndarray:
            """Calculate the variance for Bernoulli: mu * (1 - mu)."""
            return mu * (1 - mu)

        # Define functions for the Poisson family:
        def poisson_mu(eta: float | np.ndarray) -> float | np.ndarray:
            """Calculate the log link: mu = exp(eta)."""
            return np.exp(eta)

        def poisson_dmu_deta(eta: float | np.ndarray) -> float | np.ndarray:
            """Calculate the derivative of exp: dmu/deta = exp(eta) = mu."""
            return np.exp(eta)

        def poisson_V(mu: float | np.ndarray) -> float | np.ndarray:
            """Calculate the variance for Poisson: V(mu) = mu."""
            return mu

        if family == "gaussian":
            return gaussian_mu, gaussian_dmu_deta, gaussian_V
        elif family == "binomial":
            return binomial_mu, binomial_dmu_deta, binomial_V
        elif family == "poisson":
            return poisson_mu, poisson_dmu_deta, poisson_V
        else:
            raise ValueError("Unsupported family. Choose from 'gaussian', 'binomial', 'poisson'.")
