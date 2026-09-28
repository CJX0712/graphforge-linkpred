"""链接预测启发式基线（无训练打分器）。

包含 Common Neighbors / Adamic-Adar / Jaccard / Preferential Attachment /
Resource Allocation / Katz / Spectral(rank-k SVD) / Random 共 8 个基线。

作者: 晨星
"""

from __future__ import annotations

import numpy as np

from ..core.errors import HeuristicError
from ..core.types import GraphData


# --------------------------------------------------------------------- 工具
def _spectral_radius(adj: np.ndarray, iters: int = 60) -> float:
    """幂迭代估计谱半径（用于确定 Katz 阻尼系数上界）。"""
    n = adj.shape[0]
    x = np.full(n, 1.0 / np.sqrt(max(1, n)), dtype=np.float64)
    radius = 1.0
    for _ in range(iters):
        y = adj @ x
        norm = float(np.linalg.norm(y))
        if norm <= 0.0:
            return 1.0
        x = y / norm
        radius = norm
    return max(radius, 1e-6)


def truncated_svd(matrix: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """截断 SVD（n 较小时用全 SVD 保证确定性，大图用随机化 SVD）。"""
    n = matrix.shape[0]
    k = int(max(1, min(k, n - 1)))
    if n <= 1500:
        u, s, vt = np.linalg.svd(matrix, full_matrices=False)
        return u[:, :k], s[:k], vt[:k, :]
    omega = np.linalg.qr(
        np.random.default_rng(0).standard_normal((n, k + 10))
    )[0][:, : k + 10]
    y = matrix @ omega
    q, _ = np.linalg.qr(y)
    small = q.T @ matrix
    ub, sb, vbt = np.linalg.svd(small, full_matrices=False)
    return (q @ ub)[:, :k], sb[:k], vbt[:k, :]


def low_rank_reconstruction(matrix: np.ndarray, k: int) -> tuple[np.ndarray, float]:
    """秩-k 最优（Eckart–Young）重构，返回 (A_k, Frobenius 误差)。"""
    u, s, vt = truncated_svd(matrix, k)
    approx = (u * s) @ vt
    return approx, float(np.linalg.norm(matrix - approx))


def random_projection_reconstruction(
    matrix: np.ndarray, k: int, rng: np.random.Generator
) -> tuple[np.ndarray, float]:
    """秩-k 随机投影重构基线（用于证明截断 SVD 的最优性）。"""
    n = matrix.shape[0]
    k = int(max(1, min(k, n - 1)))
    gaussian = rng.standard_normal((n, k))
    q, _ = np.linalg.qr(gaussian)
    approx = q @ (q.T @ matrix)
    return approx, float(np.linalg.norm(matrix - approx))


# ----------------------------------------------------------------- 打分器
class _BaseScorer:
    name = "base"

    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs
        self.graph: GraphData | None = None
        self.n: int = 0
        self.degrees: np.ndarray | None = None

    def fit(self, graph: GraphData) -> "_BaseScorer":
        self.graph = graph
        self.n = graph.n_nodes
        self.degrees = graph.degrees
        self._prepare()
        return self

    def _prepare(self) -> None:  # pragma: no cover - 默认无额外预处理
        return None

    def _check(self, edges: np.ndarray) -> None:
        if self.graph is None:
            raise HeuristicError("打分器尚未 fit", code="E304", name=self.name)
        edges = np.asarray(edges, dtype=np.int64)
        if edges.size and (edges.max() >= self.n or edges.min() < 0):
            raise HeuristicError("边索引越界", code="E304", name=self.name)

    def score(self, edges: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class CommonNeighbors(_BaseScorer):
    name = "common_neighbors"

    def score(self, edges: np.ndarray) -> np.ndarray:
        self._check(edges)
        assert self.graph is not None
        adj = self.graph.adjacency > 0
        inter = np.asarray(adj[edges[:, 0]]) & np.asarray(adj[edges[:, 1]])
        return inter.sum(axis=1).astype(np.float64)


class AdamicAdar(_BaseScorer):
    name = "adamic_adar"

    def _prepare(self) -> None:
        assert self.degrees is not None
        safe = np.maximum(self.degrees, 2.0)
        self._inv_log_degree = 1.0 / np.log(safe)

    def score(self, edges: np.ndarray) -> np.ndarray:
        self._check(edges)
        assert self.graph is not None
        adj = self.graph.adjacency > 0
        inter = (np.asarray(adj[edges[:, 0]]) & np.asarray(adj[edges[:, 1]])).astype(np.float64)
        return inter @ self._inv_log_degree


class ResourceAllocation(_BaseScorer):
    name = "resource_allocation"

    def _prepare(self) -> None:
        assert self.degrees is not None
        self._inv_degree = 1.0 / np.maximum(self.degrees, 1.0)

    def score(self, edges: np.ndarray) -> np.ndarray:
        self._check(edges)
        assert self.graph is not None
        adj = self.graph.adjacency > 0
        inter = (np.asarray(adj[edges[:, 0]]) & np.asarray(adj[edges[:, 1]])).astype(np.float64)
        return inter @ self._inv_degree


class Jaccard(_BaseScorer):
    name = "jaccard"

    def score(self, edges: np.ndarray) -> np.ndarray:
        self._check(edges)
        assert self.graph is not None and self.degrees is not None
        adj = self.graph.adjacency > 0
        inter = (np.asarray(adj[edges[:, 0]]) & np.asarray(adj[edges[:, 1]])).sum(axis=1)
        union = self.degrees[edges[:, 0]] + self.degrees[edges[:, 1]] - inter
        return np.divide(
            inter.astype(np.float64), np.where(union > 0, union, 1.0)
        )


class PreferentialAttachment(_BaseScorer):
    name = "preferential_attachment"

    def score(self, edges: np.ndarray) -> np.ndarray:
        self._check(edges)
        assert self.degrees is not None
        return self.degrees[edges[:, 0]] * self.degrees[edges[:, 1]]


class Katz(_BaseScorer):
    """Katz 指标：K = (I - beta*A)^(-1) - I，打分取 K[u, v]。"""

    name = "katz"

    def __init__(self, beta: float | None = None, damping_ratio: float = 0.5) -> None:
        super().__init__(beta=beta, damping_ratio=damping_ratio)
        self.beta = beta
        self.damping_ratio = damping_ratio

    def _prepare(self) -> None:
        assert self.graph is not None
        adj = self.graph.adjacency
        radius = _spectral_radius(adj)
        beta = self.beta if self.beta is not None else self.damping_ratio / radius
        if not (0.0 < beta < 1.0 / radius):
            raise HeuristicError(
                "Katz beta 不满足收敛条件", code="E304", beta=beta, bound=1.0 / radius
            )
        self.beta_effective = float(beta)
        n = self.n
        system = np.eye(n, dtype=np.float64) - beta * adj
        try:
            self._kernel = np.linalg.solve(system, beta * adj)
        except np.linalg.LinAlgError as exc:  # pragma: no cover - 数值退化路径
            raise HeuristicError("Katz 线性系统奇异", code="E304") from exc

    def score(self, edges: np.ndarray) -> np.ndarray:
        self._check(edges)
        return self._kernel[edges[:, 0], edges[:, 1]]


class Spectral(_BaseScorer):
    """谱方法：邻接矩阵秩-k 重构后取 A_hat[u, v]。"""

    name = "spectral"

    def __init__(self, dim: int = 64) -> None:
        super().__init__(dim=dim)
        self.dim = int(dim)

    def _prepare(self) -> None:
        assert self.graph is not None
        approx, _ = low_rank_reconstruction(self.graph.adjacency, self.dim)
        self._approx = approx

    def score(self, edges: np.ndarray) -> np.ndarray:
        self._check(edges)
        return self._approx[edges[:, 0], edges[:, 1]]


class RandomScore(_BaseScorer):
    """随机打分基线（理论下限参照）。"""

    name = "random"

    def __init__(self, seed: int = 0) -> None:
        super().__init__(seed=seed)
        self.seed = int(seed)

    def _prepare(self) -> None:
        rng = np.random.default_rng(self.seed)
        self._values = rng.random(self.n)

    def score(self, edges: np.ndarray) -> np.ndarray:
        self._check(edges)
        return self._values[edges[:, 0]] * self._values[edges[:, 1]]


HEURISTICS = {
    "common_neighbors": CommonNeighbors,
    "adamic_adar": AdamicAdar,
    "resource_allocation": ResourceAllocation,
    "jaccard": Jaccard,
    "preferential_attachment": PreferentialAttachment,
    "katz": Katz,
    "spectral": Spectral,
    "random": RandomScore,
}


def available_heuristics() -> tuple[str, ...]:
    return tuple(sorted(HEURISTICS))


def score_edges(name: str, graph: GraphData, edges: np.ndarray, **kwargs: object) -> np.ndarray:
    """便捷入口：构造 -> fit -> score。"""
    key = str(name).strip().lower()
    if key not in HEURISTICS:
        raise HeuristicError(
            f"未知启发式: {name}，可选 {available_heuristics()}", code="E304", name=name
        )
    scorer = HEURISTICS[key](**kwargs)
    return scorer.fit(graph).score(np.asarray(edges, dtype=np.int64))
