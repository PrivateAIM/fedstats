import pprint
import numpy as np
import statsmodels.api as sm
from fedstats import FederatedGLM, PartialFisherScoring
from fedstats.util import simulate_logistic_regression

np.random.seed(42)


def fit_model_logistic_federated(X, y, max_iter=100):
    glm = FederatedGLM()

    # init local models using default of 5 clients
    local_states = [PartialFisherScoring(X[k], y[k], "binomial", fit_intercept=False) for k in range(5)]
    for i in range(max_iter):
        # update local models, retrieve them and aggregate them
        res = list(map(lambda state: state.calc_fisher_scoring_parts(), local_states))
        b_old = glm.get_coefs()
        glm.set_node_results(res)
        glm.aggregate_results()
        b_new = glm.get_coefs()
        converged = glm.check_convergence(b_old, b_new)
        if converged:
            print("Converged")
            break
        # send new coefs to local models
        for state in local_states:
            state.set_coefs(b_new)
    return glm.get_summary()


def fit_full_comparison_model(X, y):
    X_full = np.concatenate(X)
    y_full = np.concatenate(y)
    mod_fit = sm.GLM(y_full, X_full, family=sm.families.Binomial()).fit()
    return dict(coef=mod_fit.params, se=mod_fit.bse, z=mod_fit.tvalues, p=mod_fit.pvalues)


def main():
    X, y = simulate_logistic_regression(random_state=42, n=500)
    print("=== Results on full data ===")
    pprint.pprint(fit_full_comparison_model(X, y))
    print("\n \n=== Results federated ===")
    pprint.pprint(fit_model_logistic_federated(X, y))


if __name__ == "__main__":
    main()
