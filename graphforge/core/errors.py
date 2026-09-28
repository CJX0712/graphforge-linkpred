"""GraphForge 统一错误体系。

编码约定：
    E0xx  基类
    E1xx  core / config
    E2xx  data
    E3xx  graph
    E4xx  preprocess / hpo / training
    E5xx  eval / pipeline

作者: 晨星
"""

from __future__ import annotations

from typing import Any


class GraphForgeError(Exception):
    """所有 GraphForge 异常的基类，携带可检索的错误码。"""

    default_code = "E000"

    def __init__(self, message: str, code: str | None = None, **context: Any) -> None:
        self.code = code or self.default_code
        self.message = message
        self.context = context
        full = f"[{self.code}] {message}"
        if context:
            detail = ", ".join(f"{k}={v}" for k, v in sorted(context.items()))
            full = f"{full} ({detail})"
        super().__init__(full)


class ConfigError(GraphForgeError):
    """配置缺失 / 类型错误 / 越界。"""

    default_code = "E100"


class ValidationError(ConfigError):
    """schema 校验失败。"""

    default_code = "E101"


class DataError(GraphForgeError):
    """数据层异常。"""

    default_code = "E200"


class GraphBuildError(DataError):
    """图构造 / 载入失败。"""

    default_code = "E201"


class SplitError(DataError):
    """正负样本划分失败。"""

    default_code = "E202"


class GraphError(GraphForgeError):
    """图算法层异常。"""

    default_code = "E300"


class WalkError(GraphError):
    """随机游走采样失败。"""

    default_code = "E301"


class EmbeddingError(GraphError):
    """嵌入训练失败。"""

    default_code = "E302"


class EdgeOperatorError(GraphError):
    """边特征算子失败。"""

    default_code = "E303"


class HeuristicError(GraphError):
    """启发式打分失败。"""

    default_code = "E304"


class TrainingError(GraphForgeError):
    """训练链路异常。"""

    default_code = "E400"


class HPOError(TrainingError):
    """超参搜索失败。"""

    default_code = "E401"


class ConvergenceError(TrainingError):
    """训练未收敛 / 数值发散。"""

    default_code = "E402"


class BackendError(TrainingError):
    """可选后端缺失且无法降级。"""

    default_code = "E403"


class EvaluationError(GraphForgeError):
    """评测 / 流水线异常。"""

    default_code = "E500"


class MetricError(EvaluationError):
    """指标计算失败（如单类别标签）。"""

    default_code = "E501"


class PipelineError(EvaluationError):
    """流水线执行失败。"""

    default_code = "E502"


class BenchmarkError(EvaluationError):
    """基准对比失败。"""

    default_code = "E503"
