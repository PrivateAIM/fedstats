# Federated statistics WP1
![CI](https://github.com/RaphaelRe/fedstats/actions/workflows/ci.yml/badge.svg)

This repo is a collection of code that should be integrated into FLAME in the future.

## Structure

- `examples` contains examples. Currently only aggregation via meta analysis is implemented.
- `fedstats` contains the code.
  - `aggregation` module with for aggregation methods.

  Inherits from abstract class in `aggregator.py` 

  Currently, implemented methods are:
    - Meta analysis aggregation -> Weighting the local results by it's variance.
    - Average aggregation -> Weighting by number of samples (no confidence intervals)


## Contribute

- The repo uses poetry. Assuming you have installed poetry: Clone the repo and type `poetry install`.
- Activate the environment with `poetry shell`.
