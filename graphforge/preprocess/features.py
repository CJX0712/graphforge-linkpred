"""边特征装配：嵌入算子特征 + 度特征 + 可选启发式特征。

防泄漏纪律：
  * 启发式特征全部基于 train_graph（已物理删除 valid/test 边）计算；
  * 标准化器只在 train 折上 fit，valid/test 只做 transform。

作者: 晨星
"""

from __future__ import annotations

import numpy as np

from ..core.errors import TrainingError
from ..core.types import GraphData
from ..graph.edges import apply_edge_operator

DEFAULT_HEURISTIC_FEATURES = (
    "common_neighbors",
    "adamic_adar",
    "jaccard",
    "preferential_attachment",
)


class Standardizer:
    """列标准化；fit 只在训练折调用。"""

    def __init__(self, eps: float = 1e-8) -> None:
        self.eps = float(eps)
        self.mean: np.ndarray | None = None
        self.scale: np.ndarray | None = None

    def fit(self, x: np.ndarray) -> "Standardizer":
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 2 or x.shape[0] == 0:
            raise TrainingError("标准化输入必须是非空二维矩阵", code="E400")
        self.mean = x.mean(axis=0)
        std = x.std(axis=0)
        self.scale = np.where(std > self.eps, std, 1.0)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean is None or self.scale is None:
            raise TrainingError("Standardizer 尚未 fit", code="E400")
        return (np.asarray(x, dtype=np.float64) - self.mean) / self.scale

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        return self.fit(x).transform(x)


class FeatureAssembler:
    """把节点嵌入 + 图结构信息装配为边特征矩阵。"""

    def __init__(
        self,
        operator: str = "hadamard",
        use_heuristic_features: bool = False,
        heuristic_names: tuple[str, ...] = DEFAULT_HEURISTIC_FEATURES,
        standardize: bool = True,
    ) -> None:
        self.operator = operator
        self.use_heuristic_features = bool(use_heuristic_features)
        self.heuristic_names = tuple(heuristic_names)
        self.standardize = bool(standardize)
        self.graph: GraphData | None = None
        self.scaler = Standardizer()
        self._log_degree: np.ndarray | None = None
        self._scorers: list[tuple[str, object]] = []
        self._fitted = False

    # ------------------------------------------------------------------ fit
    def fit(self, graph: GraphData) -> "FeatureAssembler":
        """只用训练图拟合结构侧统计量。"""
        self.graph = graph
        self._log_degree = np.log1p(graph.degrees)
        self._scorers = []
        if self.use_heuristic_features:
            from ..graph.heuristics import HEURISTICS

            for name in self.heuristic_names:
                if name not in HEURISTICS:
                    raise TrainingError(f"未知启发式特征: {name}", code="E400", name=name)
                self._scorers.append((name, HEURISTICS[name]().fit(graph)))
        self._fitted = True
        return self

    # ------------------------------------------------------------- transform
    def _raw(self, matrix: np.ndarray, edges: np.ndarray) -> np.ndarray:
        if not self._fitted or self.graph is None or self._log_degree is None:
            raise TrainingError("FeatureAssembler 尚未 fit", code="E400")
        edges = np.asarray(edges, dtype=np.int64)
        blocks = [apply_edge_operator(matrix, edges, self.operator)]
        deg_block = np.stack(
            [
                self._log_degree[edges[:, 0]],
                self._log_degree[edges[:, 1]],
                self._log_degree[edges[:, 0]] * self._log_degree[edges[:, 1]],
            ],
            axis=1,
        )
        blocks.append(deg_block)
        for _name, scorer in self._scorers:
            values = np.asarray(scorer.score(edges), dtype=np.float64).reshape(-1, 1)
            blocks.append(np.log1p(np.maximum(values, 0.0)))
        return np.concatenate(blocks, axis=1).astype(np.float64)

    def fit_transform_train(self, matrix: np.ndarray, train_edges: np.ndarray) -> np.ndarray:
        """在训练折上拟合标准化器并变换（唯一允许 fit 标准化器的入口）。"""
        raw = self._raw(matrix, train_edges)
        if not self.standardize:
            return raw
        return self.scaler.fit_transform(raw)

    def transform(self, matrix: np.ndarray, edges: np.ndarray) -> np.ndarray:
        raw = self._raw(matrix, edges)
        if not self.standardize:
            return raw
        return self.scaler.transform(raw)

    def feature_dim(self, embedding_dim: int) -> int:
        """给定嵌入维度，返回装配后的特征维度（用于自测与文档）。"""
        edge_dim = embedding_dim * 2 if self.operator == "concat" else embedding_dim
        return edge_dim + 3 + len(self.heuristic_names if self.use_heuristic_features else ())


def heuristic_augmented_dim(config_operator: str, dim: int, n_heuristics: int) -> int:
    """便捷函数：计算装配后特征维度。"""
    edge_dim = dim * 2 if config_operator == "concat" else dim
    return edge_dim + 3 + n_heuristics
