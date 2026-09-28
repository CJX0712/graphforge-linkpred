"""node2vec 二阶偏置随机游走（Grover & Leskovec, KDD 2016）。

转移规则：从 prev --t--> cur 走到候选节点 x 时，未归一化权重为
    alpha(prev, x) * w(cur, x)
其中
    alpha = 1/p   if x == prev              （回退，d(prev,x) = 0）
            1     if (prev, x) 有边         （同距，d = 1）
            1/q   otherwise                 （外扩，d = 2）

实现要点：所有 walker 同步走一步（向量化），步数与 walker 数解耦，
保证 800~1500 节点规模下端到端仍在秒级完成。

作者: 晨星
"""

from __future__ import annotations

import numpy as np

from ..core.errors import WalkError
from ..core.types import GraphData


class Node2VecWalker:
    """二阶偏置随机游走采样器（向量化）。"""

    def __init__(self) -> None:
        self.adjacency: np.ndarray | None = None
        self.n_nodes: int = 0
        self.max_degree: int = 0
        self.nbr_ids: np.ndarray | None = None
        self.nbr_weights: np.ndarray | None = None
        self.nbr_mask: np.ndarray | None = None
        self._fitted = False

    # ------------------------------------------------------------------ fit
    def fit(self, graph: GraphData) -> "Node2VecWalker":
        adj = np.asarray(graph.adjacency, dtype=np.float64).copy()
        n = adj.shape[0]
        np.fill_diagonal(adj, 0.0)
        binary = adj > 0
        degrees = binary.sum(axis=1)
        if n == 0:
            raise WalkError("空图无法构造游走表", code="E301")
        max_deg = int(degrees.max())
        if max_deg == 0:
            raise WalkError("图中所有节点度数为 0，无法游走", code="E301")

        rows, cols = np.nonzero(binary)
        counts = np.bincount(rows, minlength=n)
        offsets = np.zeros(n + 1, dtype=np.int64)
        np.cumsum(counts, out=offsets[1:])
        positions = np.arange(rows.size, dtype=np.int64) - offsets[rows]

        nbr_ids = np.zeros((n, max_deg), dtype=np.int64)
        nbr_weights = np.zeros((n, max_deg), dtype=np.float64)
        nbr_mask = np.zeros((n, max_deg), dtype=bool)
        nbr_ids[rows, positions] = cols
        nbr_weights[rows, positions] = adj[rows, cols]
        nbr_mask[rows, positions] = True

        self.adjacency = adj
        self.n_nodes = n
        self.max_degree = max_deg
        self.nbr_ids = nbr_ids
        self.nbr_weights = nbr_weights
        self.nbr_mask = nbr_mask
        self._fitted = True
        return self

    def _require_fit(self) -> None:
        if not self._fitted or self.nbr_ids is None:
            raise WalkError("walker 尚未 fit", code="E301")

    @property
    def degrees(self) -> np.ndarray:
        self._require_fit()
        return np.asarray((self.adjacency > 0).sum(axis=1), dtype=np.float64)

    # --------------------------------------------------------------- 概率表
    def transition_weights(
        self,
        cur: np.ndarray,
        prev: np.ndarray,
        p: float,
        q: float,
        first_step: bool = False,
    ) -> np.ndarray:
        """返回 (W, D) 未归一化转移权重。

        first_step=True 时退化为「按边权的均匀游走」（node2vec 的第一步）。
        """
        self._require_fit()
        assert self.nbr_ids is not None and self.nbr_mask is not None
        assert self.nbr_weights is not None and self.adjacency is not None
        cur = np.asarray(cur, dtype=np.int64)
        prev = np.asarray(prev, dtype=np.int64)
        if cur.shape != prev.shape:
            raise WalkError("cur / prev 形状必须一致", code="E301")

        cand = self.nbr_ids[cur]                  # (W, D)
        valid = self.nbr_mask[cur]                # (W, D)
        weight = self.nbr_weights[cur] * valid    # (W, D)

        if first_step:
            return weight

        prev_safe = np.maximum(prev, 0)
        connected = self.adjacency[prev_safe[:, None], cand] > 0.0   # (W, D)
        is_prev = (cand == prev[:, None]) & valid
        alpha = np.where(is_prev, 1.0 / p, np.where(connected, 1.0, 1.0 / q))
        return alpha * weight

    def transition_probabilities(
        self,
        cur: np.ndarray,
        prev: np.ndarray,
        p: float,
        q: float,
        first_step: bool = False,
    ) -> np.ndarray:
        """归一化后的转移概率矩阵 (W, D)，每行求和为 1（孤立行全 0）。"""
        raw = self.transition_weights(cur, prev, p, q, first_step=first_step)
        total = raw.sum(axis=1, keepdims=True)
        return np.divide(raw, np.where(total > 0.0, total, 1.0))

    # ---------------------------------------------------------------- 采样
    def sample(
        self,
        rng: np.random.Generator,
        num_walks: int,
        walk_length: int,
        p: float,
        q: float,
    ) -> np.ndarray:
        """返回形状 (n_nodes * num_walks, walk_length) 的 int64 游走序列。"""
        self._require_fit()
        if p <= 0 or q <= 0:
            raise WalkError("p / q 必须为正", code="E301", p=p, q=q)
        if num_walks < 1:
            raise WalkError("num_walks 必须 >= 1", code="E301", num_walks=num_walks)
        if walk_length < 2:
            raise WalkError("walk_length 必须 >= 2", code="E301", walk_length=walk_length)

        n = self.n_nodes
        n_walkers = n * int(num_walks)

        starts = np.tile(np.arange(n, dtype=np.int64), int(num_walks))
        cur = starts.copy()
        prev = np.full(n_walkers, -1, dtype=np.int64)
        walks = np.empty((n_walkers, int(walk_length)), dtype=np.int64)
        walks[:, 0] = cur

        for step in range(1, int(walk_length)):
            raw = self.transition_weights(cur, prev, p, q, first_step=(step == 1))
            cumulative = np.cumsum(raw, axis=1)
            total = cumulative[:, -1]
            threshold = rng.random(n_walkers) * total
            choice = (cumulative < threshold[:, None]).sum(axis=1)
            choice = np.clip(choice, 0, self.max_degree - 1)
            assert self.nbr_ids is not None
            nxt = self.nbr_ids[cur, choice]
            dead_end = total <= 0.0
            nxt = np.where(dead_end, cur, nxt)
            prev = cur
            cur = nxt
            walks[:, step] = cur

        return walks


def deepwalk_walks(
    walker: Node2VecWalker,
    rng: np.random.Generator,
    num_walks: int,
    walk_length: int,
) -> np.ndarray:
    """DeepWalk：p = q = 1（退化为按边权的均匀游走）。"""
    return walker.sample(rng, num_walks, walk_length, p=1.0, q=1.0)
