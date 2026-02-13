import numpy as np
import statsmodels.api as sm


class LocalLinearRegression:
    """
    Wraps a standard regression such that it can directly send results to an aggregator unit.
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        fit_intercept: bool = True,
        standardize: bool = False,
        **kwargs,
    ) -> None:
        self.X = self.make_covariate_matrix(X=X, fit_intercept=fit_intercept, standardize=standardize)
        self.y = y
        self.n, self.p = X.shape
        self.model = sm.OLS(self.y, self.X, **kwargs)
        self.model_fitted = False

    def make_covariate_matrix(self, X, fit_intercept: bool, standardize: bool = False) -> np.ndarray:
        """
        Function to build a covariate matrix.

        Right now, it only adds intercept. Later, it can also standardize or make splines.
        """
        if standardize:
            raise NotImplementedError("Not implemented yet. Do it manually beforehand.")
        if fit_intercept:
            X = np.c_[np.ones(X.shape[0]), X]
        return X

    def fit(self, *args, **kwargs) -> None:
        """
        Wraps the fit function from statsmodels.api.sm.OLS
        """
        print("Fitting model...")
        self.result = self.model.fit(*args, **kwargs)
        self.model_fitted = True
        print("Done!")

    def get_result(self, use_sd: bool = True, use_n: bool = False) -> list[tuple]:
        """
        Get the results such that they can be processes by an aggregator.

        Returns the estiamtors and either the variance or the sample size.
        """
        if not use_sd and not use_n:
            raise Exception("Either use_sd or use_n has to be True.")
        if self.model_fitted:
            coefs = self.result.params
        else:
            raise Exception("Fit model first.")
        if use_sd:
            weighter = self.result.bse**2
        if use_n:
            weighter = np.ones(coefs.shape, dtype=int) * self.n

        return list(zip(coefs, weighter))  # type: ignore
