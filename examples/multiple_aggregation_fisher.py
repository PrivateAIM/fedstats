from fedstats import FisherAggregation, PartialFisherScoring

import numpy as np
import pandas as pd

def MyPartialFisherScoring(X, y, family, tol=1e-6, max_iter=100, verbose=False):
    model = PartialFisherScoring(X, y, family)
    
    # Iterative procedure to update beta
    for _ in range(max_iter):
        fisher_info, rhs = model.calc_fisher_scoring_parts()
        delta = np.linalg.solve(fisher_info, rhs)
        model.beta += delta
        if verbose:
            print("Current beta coefficients:", model.beta)
        if np.linalg.norm(delta) < tol:
            break

    # Compute standard errors from the Fisher information matrix
    se = np.sqrt(np.diag(np.linalg.inv(fisher_info)))
    
    if verbose:
        print("Final beta coefficients:", model.beta)
        print("Final Fisher information matrix:\n", fisher_info)
        print("Computed standard errors:", se)
    
    return model.beta, se

def load_titanic_data():
    url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    data = pd.read_csv(url)

    # Keep only the relevant columns: target 'Survived' and selected predictors.
    data = data[['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']]

    data = data.dropna()

    data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})

    return data

def load_split_data(num_clients=3, random_state=42):
    data = load_titanic_data()
    data = data.sample(frac=1, random_state=random_state).reset_index(drop=True)

    chunk_size = len(data) // num_clients
    remainder = len(data) % num_clients

    sub_datasets = []

    start_idx = 0
    for i in range(num_clients):
        end_idx = start_idx + chunk_size + (1 if i < remainder else 0)
        sub_datasets.append(data.iloc[start_idx:end_idx])
        start_idx = end_idx

    return sub_datasets

def apply_fisher(sub_datasets, verbose=False, return_local=False):
    local_results = []
    for df in sub_datasets:
        # For Titanic: target 'Survived' and predictors: 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare'
        y = df['Survived'].values
        X = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']].values
        beta, se = MyPartialFisherScoring(X=X, y=y, family='binomial', verbose=verbose)
        local_results.append((beta, se))

    aggregator = FisherAggregation(node_results=local_results)
    aggregator.aggregate_results(verbose=verbose)
    combined_p = aggregator.get_aggregated_results()
    # combined_p = aggregator.aggregate(local_results, verbose=verbose)

    return (combined_p, local_results) if return_local else combined_p

if __name__ == '__main__':
    # Split the Titanic data into 2 parts.
    sub_datasets = load_split_data(num_clients=2, random_state=42)

    # Get aggregated p‑values and local (beta, se) pairs.
    combined_p, local_results = apply_fisher(sub_datasets, verbose=False, return_local=True)

    # Extract predictor names from one sub‐dataset and prepend 'Intercept'.
    predictor_cols = list(sub_datasets[0].drop(columns=['Survived']).columns)
    coef_names = ['Intercept'] + predictor_cols

    # For each predictor, apply the meta‑analysis aggregator separately.
    import numpy as np
    from fedstats.aggregation.meta_analysis import MetaAnalysisAggregator

    n_coeff = local_results[0][0].shape[0]  # number of coefficients (should equal len(coef_names))
    aggregated_betas = []
    aggregated_vars = []

    for j in range(n_coeff):
        local_results_j = [(beta[j], se[j]) for beta, se in local_results]
        meta_agg_unit = MetaAnalysisAggregator(node_results=local_results_j)
        meta_agg_unit.aggregate_results()
        agg_results = meta_agg_unit.get_aggregated_results()
        aggregated_betas.append(agg_results["aggregated_results"])
        aggregated_vars.append(agg_results["aggregated_variance"])

    aggregated_betas = np.array(aggregated_betas)
    aggregated_vars = np.array(aggregated_vars)
    aggregated_se = np.sqrt(aggregated_vars)

    import pandas as pd
    results_df = pd.DataFrame({
        'Coefficient': coef_names,
        'P-value': combined_p,
        'Beta': aggregated_betas,
        'SE': aggregated_se,
        'Odds Ratio': np.exp(aggregated_betas)
    })
    
    formatters = {
        'Coefficient': lambda x: f"{x:>10}",
        'P-value': lambda x: f"{float(x):>12.3e}",
        'Beta': lambda x: f"{float(x):>10.3f}",
        'SE': lambda x: f"{float(x):>8.3f}",
        'Odds Ratio': lambda x: f"{float(x):>10.3f}"
    }

    table_str = results_df.to_string(index=False, justify="center", formatters=formatters)
    header, body = table_str.split('\n', 1)
    print(header)
    print('-' * len(header))
    print(body)

