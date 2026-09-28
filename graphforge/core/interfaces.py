"""结构化接口契约（Protocol），用于约束各模块实现。

作者: 晨星
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np

from .types import EdgeSplit, EmbeddingResult, GraphData


@runtime_checkable
class GraphSource(Protocol):
    """图数据源。"""

    def build(self, rng: np.random.Generator) -> GraphData: ...


@runtime_checkable
class WalkSampler(Protocol):
    """随机游走采样器。"""

    def fit(self, graph: GraphData) -> "WalkSampler": ...

    def sample(
        self, rng: np.random.Generator, num_walks: int, walk_length: int, p: float, q: float
    ) -> np.ndarray: ...


@runtime_checkable
class Embedder(Protocol):
    """节点嵌入器。"""

    def fit(self, graph: GraphData, walks: np.ndarray) -> EmbeddingResult: ...


@runtime_checkable
class EdgeOperator(Protocol):
    """边特征算子。"""

    name: str

    def __call__(self, matrix: np.ndarray, edges: np.ndarray) -> np.ndarray: ...


@runtime_checkable
class Classifier(Protocol):
    """下游链接预测分类器。"""

    def fit(self, x: np.ndarray, y: np.ndarray) -> "Classifier": ...

    def predict_proba(self, x: np.ndarray) -> np.ndarray: ...

    def backend(self) -> str: ...


@runtime_checkable
class Scorer(Protocol):
    """启发式打分器（无需训练）。"""

    name: str

    def fit(self, graph: GraphData) -> "Scorer": ...

    def score(self, edges: np.ndarray) -> np.ndarray: ...


@runtime_checkable
class Tuner(Protocol):
    """超参搜索器。"""

    def search(self, split: EdgeSplit, cfg: Any) -> dict[str, Any]: ...


@runtime_checkable
class Pipeline(Protocol):
    """端到端流水线。"""

    def run(self) -> dict[str, Any]: ...
