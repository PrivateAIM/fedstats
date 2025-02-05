import numpy as np
import statsmodels.api as sm
from fedstats.aggregation.fed_glm import FedGLM
from fedstats.models.local_fisher_scoring import LocalFisherScoring
from fedstats.util import simulate_logistic_regression

np.random.seed(42)


def fit_model_logistic_federated(X, y, max_iter=100):
    glm = FedGLM([])

    # init local models using default of 5 clients
    local_states = [
        LocalFisherScoring(X[k], y[k], "binomial", fit_intercept=False)
        for k in range(5)
    ]
    for i in range(max_iter):
        # update local models, retrieve them and aggregate them
        res = list(map(lambda state: state.calc_fisher_scoring_parts(), local_states))
        b_old = glm.get_coefs()
        glm.set_results(res)
        glm.aggregate_results()
        b_new = glm.get_coefs()
        converged = glm.check_convergence(b_old, b_new)
        if converged:
            print("Converged")
            break
        # send new coefs to local models
        for state in local_states:
            state.set_coefs(b_new)
    return glm.get_coefs()


def fit_full_comparison_model(X, y):
    X_full = np.concatenate(X)
    y_full = np.concatenate(y)
    mod_fit = sm.GLM(y_full, X_full, family=sm.families.Binomial()).fit()
    return mod_fit.params


def main():
    X, y = simulate_logistic_regression(500)
    print("Coefs on full data:", fit_full_comparison_model(X, y))
    print("Estimated coefficients:", fit_model_logistic_federated(X, y))
