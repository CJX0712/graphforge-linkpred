"""正负样本划分（严格防泄漏）。

不变量：
  1. train_graph 已物理删除 valid + test 正边（不是掩码，是真的没有这条边）；
  2. 训练前嵌入只在 train_graph 上跑；
  3. 所有负样本均不属于「全集正边集合」，不会把真实边误标为负类；
  4. train_graph 保持连通（生成树边强制留在训练集）。

作者: 晨星
"""

from __future__ import annotations

import numpy as np

from ..core.errors import SplitError
from ..core.types import EdgeSplit, GraphData


def _spanning_tree_mask(n: int, edges: np.ndarray) -> np.ndarray:
    """并查集：返回哪些边属于生成森林（True 表示必须留在训练集）。"""
    parent = np.arange(n, dtype=np.int64)

    def find(x: int) -> int:
        root = x
        while parent[root] != root:
            root = int(parent[root])
        while parent[x] != root:
            parent[x], x = root, int(parent[x])
        return root

    keep = np.zeros(edges.shape[0], dtype=bool)
    for idx in range(edges.shape[0]):
        u, v = int(edges[idx, 0]), int(edges[idx, 1])
        ru, rv = find(u), find(v)
        if ru != rv:
            parent[ru] = rv
            keep[idx] = True
    return keep


def _sample_negative_pairs(
    n: int,
    count: int,
    excluded_sorted: np.ndarray,
    rng: np.random.Generator,
    max_rounds: int = 200,
) -> np.ndarray:
    """采样 count 个互不重复、且不在 excluded 中的无向节点对。"""
    if count <= 0:
        return np.zeros((0, 2), dtype=np.int64)
    pool = np.zeros(0, dtype=np.int64)
    for _ in range(max_rounds):
        need = count - int(pool.size)
        if need <= 0:
            break
        batch = int(need * 1.5) + 64
        u = rng.integers(0, n, size=batch)
        v = rng.integers(0, n, size=batch)
        lo = np.minimum(u, v)
        hi = np.maximum(u, v)
        keys = lo.astype(np.int64) * n + hi.astype(np.int64)
        keys = keys[lo != hi]
        if excluded_sorted.size:
            pos = np.clip(np.searchsorted(excluded_sorted, keys), 0, excluded_sorted.size - 1)
            keys = keys[excluded_sorted[pos] != keys]
        if keys.size == 0:
            continue
        pool = np.unique(np.concatenate([pool, keys])) if pool.size else np.unique(keys)
    if pool.size < count:
        raise SplitError(
            "负样本采样不足（图过于稠密）", code="E202", need=count, got=int(pool.size)
        )
    chosen = np.sort(pool[:count])
    return np.stack([chosen // n, chosen % n], axis=1).astype(np.int64)


def split_edges(
    graph: GraphData,
    train_ratio: float = 0.7,
    valid_ratio: float = 0.1,
    test_ratio: float = 0.2,
    negative_ratio: float = 1.0,
    rng: np.random.Generator | None = None,
) -> EdgeSplit:
    """把无向图划分为 train / valid / test 的正负样本。"""
    if rng is None:
        rng = np.random.default_rng(0)
    n = graph.n_nodes
    edges = graph.edge_list()
    m = edges.shape[0]
    if m < 10:
        raise SplitError("边数过少，无法划分", code="E202", n_edges=m)

    perm = rng.permutation(m)
    shuffled = edges[perm]
    tree_mask = _spanning_tree_mask(n, shuffled)
    tree_edges = shuffled[tree_mask]
    free_edges = shuffled[~tree_mask]

    n_test = int(round(test_ratio * m))
    n_valid = int(round(valid_ratio * m))
    if free_edges.shape[0] < n_test + n_valid:
        n_test = int(free_edges.shape[0] * test_ratio / max(1e-12, test_ratio + valid_ratio))
        n_valid = free_edges.shape[0] - n_test
    test_pos = free_edges[:n_test]
    valid_pos = free_edges[n_test : n_test + n_valid]
    train_pos = np.concatenate([free_edges[n_test + n_valid :], tree_edges], axis=0)

    train_adj = np.zeros((n, n), dtype=np.float64)
    train_adj[train_pos[:, 0], train_pos[:, 1]] = 1.0
    train_adj[train_pos[:, 1], train_pos[:, 0]] = 1.0
    np.fill_diagonal(train_adj, 0.0)
    train_graph = GraphData(adjacency=train_adj, name=f"{graph.name}::train")

    # 正边全集 key（用于排除负样本命中真实边）
    all_pos = np.concatenate([train_pos, valid_pos, test_pos], axis=0)
    pos_keys = np.unique(
        np.minimum(all_pos[:, 0], all_pos[:, 1]).astype(np.int64) * n
        + np.maximum(all_pos[:, 0], all_pos[:, 1]).astype(np.int64)
    )

    def _negatives(n_pos: int) -> np.ndarray:
        return _sample_negative_pairs(
            n=n, count=int(round(negative_ratio * n_pos)), excluded_sorted=pos_keys, rng=rng
        )

    train_neg = _negatives(train_pos.shape[0])
    valid_neg = _negatives(valid_pos.shape[0])
    test_neg = _negatives(test_pos.shape[0])

    def _assemble(pos: np.ndarray, neg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x = np.concatenate([pos, neg], axis=0).astype(np.int64)
        y = np.concatenate(
            [np.ones(pos.shape[0], dtype=np.int64), np.zeros(neg.shape[0], dtype=np.int64)]
        )
        order = rng.permutation(x.shape[0])
        return x[order], y[order]

    train_edges, train_labels = _assemble(train_pos, train_neg)
    valid_edges, valid_labels = _assemble(valid_pos, valid_neg)
    test_edges, test_labels = _assemble(test_pos, test_neg)

    return EdgeSplit(
        train_graph=train_graph,
        train_edges=train_edges,
        train_labels=train_labels,
        valid_edges=valid_edges,
        valid_labels=valid_labels,
        test_edges=test_edges,
        test_labels=test_labels,
        positive_edges=all_pos,
        meta={
            "n_nodes": n,
            "n_edges_total": m,
            "n_train_pos": int(train_pos.shape[0]),
            "n_valid_pos": int(valid_pos.shape[0]),
            "n_test_pos": int(test_pos.shape[0]),
            "n_train_neg": int(train_neg.shape[0]),
            "n_valid_neg": int(valid_neg.shape[0]),
            "n_test_neg": int(test_neg.shape[0]),
        },
    )
