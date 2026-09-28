"""GraphForge 流水线层：单次运行与跨 seed 基准对比。"""

from __future__ import annotations

from .benchmark import run_benchmark
from .link_prediction import LinkPredictionPipeline

__all__ = ["LinkPredictionPipeline", "run_benchmark"]
