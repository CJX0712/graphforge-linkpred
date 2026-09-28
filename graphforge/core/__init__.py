"""GraphForge 核心层：类型、错误、配置、接口、随机性、能力探测。

调用方向：core 不依赖任何上层模块。
"""

from __future__ import annotations

from .config import GraphForgeConfig, config_from_env, load_config, validate_config
from .errors import (
    BackendError,
    BenchmarkError,
    ConfigError,
    DataError,
    EdgeOperatorError,
    EmbeddingError,
    EvaluationError,
    GraphBuildError,
    GraphError,
    GraphForgeError,
    HeuristicError,
    HPOError,
    MetricError,
    PipelineError,
    SplitError,
    TrainingError,
    WalkError,
)
from .seed import derive_seed, make_rng, set_global_seed
from .types import (
    BenchmarkResult,
    EdgeSplit,
    FailureCase,
    GraphData,
    SeedResult,
)
from .types import EmbeddingResult as EmbeddingResult  # noqa: F401 - 对外再导出

__all__ = [
    "BackendError",
    "BenchmarkError",
    "BenchmarkResult",
    "ConfigError",
    "DataError",
    "EdgeOperatorError",
    "EdgeSplit",
    "EmbeddingError",
    "EvaluationError",
    "FailureCase",
    "GraphBuildError",
    "GraphData",
    "GraphError",
    "GraphForgeConfig",
    "GraphForgeError",
    "HeuristicError",
    "HPOError",
    "MetricError",
    "PipelineError",
    "SeedResult",
    "SplitError",
    "TrainingError",
    "WalkError",
    "config_from_env",
    "derive_seed",
    "load_config",
    "make_rng",
    "set_global_seed",
    "validate_config",
]
