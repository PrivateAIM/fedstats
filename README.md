# Federated statistics WP1

![Lint](https://github.com/PrivateAIM/fedstats/actions/workflows/lint.yml/badge.svg)
![Test](https://github.com/PrivateAIM/fedstats/actions/workflows/test.yml/badge.svg)

This repository contains methods for aggregating statistics.

## Structure

- `examples` contains examples
  - `standalone` contains examples that show how fedstats can be used in general.
  - `flame` contains examples that show how fedstats can be used within FLAME.
- `fedstats` contains the code.
  - `aggregation` module is used for aggregation methods.
  - `models` statistical models that can be used with the aggregators.
- `tests` contains tests.

## Currently implemented methods

- Aggregation Methods:
  - Meta-analysis aggregation: weighting local results by their variance (also allows the calculation of confidence intervals)
  - Average aggregation: weighting local results by their number of samples
  - Aggregation of Generalized Linear Models
  - Aggregation of p values using Fisher's method

- Models
  - `LocalLinearRegression` wraps a standard linear regression model from statsmodels for direct use with the aggregators.
  - `LocalFisherScoring` implements generalized linear models with Fisher scoring.

## Contribute
- The repo uses poetry for dependency management. Assuming you have installed poetry: Clone the repo and type `poetry install`.
- Activate the environment with `poetry env activate`.
- Testing is done with pytest: `poetry run pytest tests/`.
