# Federated statistics WP1

![CI](https://github.com/PrivateAIM/fedstats/actions/workflows/ci.yml/badge.svg)

This repo is a collection of code that should be integrated into FLAME in the future.

## Structure

- `examples` contains example usage.
- `fedstats` contains the code.
  - `aggregation` module is used for aggregation methods.
  - `models` statistical models that can be used with the aggregators.

  Inherits from abstract class in `aggregator.py`
- `tests` contains tests.

## Currently, implemented methods are

- aggregation methods:
  - Meta-analysis aggregation -> Weighting the local results by its variance.
  - Average aggregation -> Weighting by number of samples (no confidence intervals).
  - Federated GLM example with logistic regression

- models
  - linear regression -> wraps standard model such that an aggregator can directly use results.

## Contribute

- The repo uses poetry. Assuming you have installed poetry: Clone the repo and type `poetry install`.
- Activate the environment with `poetry env activate`.
- Testing is done with pytest: `poetry run pytest tests/`.
