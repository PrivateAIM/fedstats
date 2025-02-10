from .aggregation.meta_analysis import MetaAnalysisAggregator as _MetaAnalysisAggregator
from .aggregation.average import AverageAggregator as _AverageAggregator
from .aggregation.fed_glm import FedGLM as _FedGLM

from .models.local_fisher_scoring import LocalFisherScoring as _LocalFisherScoring
from .models.local_linear_regression import (
    LocalLinearRegression as _LocalLinearRegression,
)


# General Aggregators
AverageAggregation = _AverageAggregator
MetaAnalysisAggregation = _MetaAnalysisAggregator

# Federated GLM
FederatedGLM = _FedGLM
PartialFisherScoring = _LocalFisherScoring

# Other
LinearRegression = _LocalLinearRegression
