from .aggregation.average import AverageAggregator as _AverageAggregator
from .aggregation.fed_glm import FedGLM as _FedGLM
from .aggregation.fisher_aggregator import FisherAggregator as _FisherAggregator
from .aggregation.meta_analysis import MetaAnalysisAggregator as _MetaAnalysisAggregator
from .models.local_fisher_scoring import LocalFisherScoring as _LocalFisherScoring
from .models.local_linear_regression import (
    LocalLinearRegression as _LocalLinearRegression,
)

# General Aggregators
AverageAggregation = _AverageAggregator
MetaAnalysisAggregation = _MetaAnalysisAggregator
FisherAggregation = _FisherAggregator

# Federated GLM
FederatedGLM = _FedGLM
PartialFisherScoring = _LocalFisherScoring

# Other
LinearRegression = _LocalLinearRegression
