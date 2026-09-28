"""GraphForge 核心数据契约（dataclass）。

作者: 晨星
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

EDGE_OPERATORS = ("hadamard", "average", "weighted_l1", "weighted_l2", "concat")


@dataclass(eq=False)
class GraphData:
    """无向图。adjacency 为 (n, n) float64 对称矩阵，无自环，0 表示无边。"""

    adjacency: np.ndarray
    name: str = "graph"

    def __post_init__(self) -> None:
        self.adjacency = np.asarray(self.adjacency, dtype=np.float64)
        if self.adjacency.ndim != 2 or self.adjacency.shape[0] != self.adjacency.shape[1]:
            raise ValueError("adjacency 必须是方阵")

    @property
    def n_nodes(self) -> int:
        return int(self.adjacency.shape[0])

    @property
    def n_edges(self) -> int:
        """无向边数（每条边只计一次）。"""
        return int((self.adjacency > 0).sum() // 2)

    @property
    def degrees(self) -> np.ndarray:
        return np.asarray((self.adjacency > 0).sum(axis=1), dtype=np.float64)

    def edge_list(self) -> np.ndarray:
        """返回 (m, 2) int64 数组，仅保留 i < j。"""
        iu = np.triu_indices(self.n_nodes, k=1)
        mask = self.adjacency[iu] > 0
        rows = iu[0][mask]
        cols = iu[1][mask]
        return np.stack([rows, cols], axis=1).astype(np.int64)

    def has_edge(self, u: int, v: int) -> bool:
        return bool(self.adjacency[int(u), int(v)] > 0)


@dataclass(eq=False)
class EdgeSplit:
    """训练 / 验证 / 测试划分。

    关键不变量（防数据泄漏）：
      * train_graph 已物理删除 valid + test 正边；
      * 所有负样本均不在「全集正边集合」内，不会把真实边误标为负类。
    """

    train_graph: GraphData
    train_edges: np.ndarray
    train_labels: np.ndarray
    valid_edges: np.ndarray
    valid_labels: np.ndarray
    test_edges: np.ndarray
    test_labels: np.ndarray
    positive_edges: np.ndarray
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(eq=False)
class EmbeddingResult:
    """嵌入训练产物。"""

    matrix: np.ndarray
    n_nodes: int
    dim: int
    params: dict[str, Any] = field(default_factory=dict)
    loss_trace: list[float] = field(default_factory=list)


@dataclass(eq=False)
class SeedResult:
    """单个 seed 上某方法的评测结果。"""

    seed: int
    method: str
    auc: float
    ap: float
    accuracy: float
    n_train: int
    n_test: int
    seconds: float
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(eq=False)
class BenchmarkResult:
    """跨 seed 聚合结果（mean ± std）。"""

    method: str
    seeds: list[int]
    auc_mean: float
    auc_std: float
    ap_mean: float
    ap_std: float
    per_seed: list[SeedResult] = field(default_factory=list)
    notes: str = ""

    def as_row(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "n_seeds": len(self.seeds),
            "auc_mean": round(self.auc_mean, 6),
            "auc_std": round(self.auc_std, 6),
            "ap_mean": round(self.ap_mean, 6),
            "ap_std": round(self.ap_std, 6),
            "notes": self.notes,
        }


@dataclass(eq=False)
class FailureCase:
    """典型误例（含归因）。"""

    kind: str  # "false_positive" | "false_negative"
    u: int
    v: int
    score: float
    label: int
    common_neighbors: int
    deg_u: int
    deg_v: int
    attribution: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "edge": [int(self.u), int(self.v)],
            "score": round(float(self.score), 6),
            "label": int(self.label),
            "common_neighbors": int(self.common_neighbors),
            "deg_u": int(self.deg_u),
            "deg_v": int(self.deg_v),
            "attribution": self.attribution,
        }
