"""Implements a local linear regression model that can be used in the federated setting."""

import numpy as np
import statsmodels.api as sm


class LocalLinearRegression:
    """Wraps a statsmodels regression such that it can directly send results to an aggregator unit."""

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        fit_intercept: bool = True,
        standardize: bool = False,
        **kwargs,
    ) -> None:
        """
        Initialize the Local Linear Regression model.

        Parameters
        ----------
            X : np.ndarray
                The feature matrix (n_samples x n_features).
            y : np.ndarray
                The response vector (n_samples,).
            fit_intercept : bool, optional
                Whether to add an intercept term to the model. Default is True.
            standardize : bool, optional
                Whether to standardize the features. Default is False. Not implemented yet.
            **kwargs
                Additional keyword arguments to pass to the statsmodels OLS constructor.
        """
        self.X = self.make_covariate_matrix(X=X, fit_intercept=fit_intercept, standardize=standardize)
        self.y = y
        self.n, self.p = X.shape
        self.model = sm.OLS(self.y, self.X, **kwargs)
        self.model_fitted = False

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
            NotImplementedError: If standardization is requested, since it's not implemented yet.
        """
        if standardize:
            raise NotImplementedError("Not implemented yet. Do it manually beforehand.")
        if fit_intercept:
            X = np.c_[np.ones(X.shape[0]), X]
        return X

    def fit(self, *args, **kwargs) -> None:
        """
        Fit the local linear regression model using statsmodels OLS.

        Stores the fitted model and results in the object.

        Parameters
        ----------
            *args, **kwargs
                Additional arguments to pass to the statsmodels OLS fit method.
        """
        print("Fitting model...")
        self.result = self.model.fit(*args, **kwargs)
        self.model_fitted = True
        print("Done!")

    def get_result(self, use_sd: bool = True, use_n: bool = False) -> list[tuple]:
        """
        Get the results in a format that can be processed by an aggregator.

        Parameters
        ----------
            use_sd : bool, optional
                Whether to use the standard errors as weights for aggregation. Default is True.
            use_n : bool, optional
                Whether to use the sample size as weights for aggregation. Default is False.

        Returns
        -------
            list[tuple]
                A list of tuples, where each tuple contains the effect size and
                the corresponding weight (either standard error or sample size).

        Raises
        ------
            ValueError: If neither use_sd nor use_n is True, or if the model has not been fitted yet.
        """
        if not use_sd and not use_n:
            raise ValueError("Either use_sd or use_n has to be True.")
        if self.model_fitted:
            coefs = self.result.params
        else:
            raise ValueError("Fit model first.")
        if use_sd:
            weighter = self.result.bse**2
        if use_n:
            weighter = np.ones(coefs.shape, dtype=int) * self.n

        return list(zip(coefs, weighter, strict=False))  # type: ignore
