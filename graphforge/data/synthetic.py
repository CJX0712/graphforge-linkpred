"""合成图生成器 + 内置真实小图。

提供三种标准基准：
  * stochastic_block_model   —— 经典 SBM（同质度、强社群）
  * lfr_graph                —— LFR 基准（幂律度分布 + 社群，networkx 实现，
                                失败时自动降级到 degree_corrected_sbm）
  * karate_club              —— 内置真实小图（Zachary 空手道俱乐部）

作者: 晨星
"""

from __future__ import annotations

import numpy as np

from ..core.capabilities import available_networkx
from ..core.errors import GraphBuildError
from ..core.types import GraphData


def _symmetrize(adj: np.ndarray) -> np.ndarray:
    adj = np.maximum(adj, adj.T)
    np.fill_diagonal(adj, 0.0)
    return adj


def _largest_component(adj: np.ndarray) -> np.ndarray:
    """保留最大连通分量，剔除孤立节点（避免游走卡死）。"""
    n = adj.shape[0]
    binary = adj > 0
    if available_networkx():
        try:  # pragma: no cover - 依赖可选后端
            import networkx as nx

            g = nx.from_numpy_array(binary)
            comps = list(nx.connected_components(g))
            if not comps:
                raise GraphBuildError("图为空")
            keep = np.zeros(n, dtype=bool)
            keep[list(max(comps, key=len))] = True
            sub = binary[np.ix_(keep, keep)]
            return sub.astype(np.float64)
        except GraphBuildError:
            raise
        except Exception:  # noqa: BLE001 - 后端异常时走纯 numpy 路径
            pass

    # 纯 numpy BFS 兜底
    seen = np.zeros(n, dtype=bool)
    best: list[int] = []
    for start in range(n):
        if seen[start]:
            continue
        stack = [start]
        comp: list[int] = []
        seen[start] = True
        while stack:
            node = stack.pop()
            comp.append(node)
            nbrs = np.nonzero(binary[node])[0]
            for nb in nbrs:
                if not seen[nb]:
                    seen[nb] = True
                    stack.append(int(nb))
        if len(comp) > len(best):
            best = comp
    if not best:
        raise GraphBuildError("图为空，无法提取连通分量")
    keep = np.zeros(n, dtype=bool)
    keep[np.asarray(sorted(best), dtype=np.int64)] = True
    return binary[np.ix_(keep, keep)].astype(np.float64)


def stochastic_block_model(
    n: int,
    n_communities: int = 4,
    p_in: float = 0.06,
    p_out: float = 0.012,
    rng: np.random.Generator | None = None,
    name: str = "sbm",
) -> GraphData:
    """经典随机块模型。社群内连边概率 p_in，社群间 p_out。"""
    if n <= 0:
        raise GraphBuildError("节点数必须为正", code="E201", n=n)
    if rng is None:
        rng = np.random.default_rng(0)
    labels = np.arange(n) % max(1, n_communities)
    same = labels[:, None] == labels[None, :]
    prob = np.where(same, p_in, p_out).astype(np.float64)
    np.fill_diagonal(prob, 0.0)
    draws = rng.random((n, n))
    upper = np.triu(draws < prob, k=1)
    adj = (upper | upper.T).astype(np.float64)
    adj = _largest_component(adj)
    if adj.shape[0] < 10:
        raise GraphBuildError("SBM 生成后连通分量过小", code="E201", size=adj.shape[0])
    return GraphData(adjacency=adj, name=name)


def sbm_from_avg_degree(
    n: int,
    n_communities: int = 4,
    avg_degree: float = 10.0,
    homophily: float = 10.0,
    rng: np.random.Generator | None = None,
    name: str = "sbm",
) -> GraphData:
    """按「目标平均度 + 同配比 homophily = p_in / p_out」反解 SBM 参数。

    这样 avg_degree 与 homophily 都是可直接解释的量，而不是两个凭感觉填的概率。
    """
    if rng is None:
        rng = np.random.default_rng(0)
    comm = max(1, int(n_communities))
    intra = max(1.0, n / comm - 1.0)
    inter = max(1.0, n - n / comm)
    p_out = float(avg_degree) / (intra * float(homophily) + inter)
    p_in = p_out * float(homophily)
    p_in = float(min(p_in, 0.9))
    return stochastic_block_model(
        n=n, n_communities=comm, p_in=p_in, p_out=p_out, rng=rng, name=name
    )


def degree_corrected_sbm(
    n: int,
    n_communities: int = 5,
    tau: float = 2.2,
    lambda_in: float = 6.0,
    lambda_out: float = 1.0,
    rng: np.random.Generator | None = None,
    name: str = "dc-sbm",
) -> GraphData:
    """度修正 SBM：幂律度分布 + 社群结构（Chung-Lu 风格）。

    用作 LFR 生成失败时的确定性兜底，本身也是标准基准图。
    """
    if rng is None:
        rng = np.random.default_rng(0)
    labels = np.arange(n) % max(1, n_communities)
    # Pareto 权重 -> 幂律度分布
    weights = (1.0 - rng.random(n)) ** (-1.0 / (tau - 1.0))
    weights = weights / weights.mean()
    same = labels[:, None] == labels[None, :]
    lam = np.where(same, lambda_in, lambda_out)
    prob = np.outer(weights, weights) * lam
    scale = float(np.triu(prob, k=1).sum()) / (n * (n - 1) / 2)
    if scale > 0.25:
        prob = prob * (0.25 / scale)
    prob = np.clip(prob, 0.0, 1.0)
    np.fill_diagonal(prob, 0.0)
    draws = rng.random((n, n))
    upper = np.triu(draws < prob, k=1)
    adj = (upper | upper.T).astype(np.float64)
    adj = _largest_component(adj)
    if adj.shape[0] < 10:
        raise GraphBuildError("度修正 SBM 连通分量过小", code="E201", size=adj.shape[0])
    return GraphData(adjacency=adj, name=name)


def lfr_graph(
    n: int,
    tau1: float = 2.5,
    tau2: float = 1.5,
    mu: float = 0.35,
    avg_degree: float = 10.0,
    max_degree: int | None = None,
    min_community: int = 20,
    seed: int = 0,
    rng: np.random.Generator | None = None,
) -> GraphData:
    """LFR 基准图；networkx 缺失或反复失败时降级为度修正 SBM。"""
    if rng is None:
        rng = np.random.default_rng(seed)
    if max_degree is None:
        max_degree = max(int(avg_degree * 4), int(n**0.5))
    if not available_networkx():
        return degree_corrected_sbm(n=n, rng=rng, name="dc-sbm(lfr-fallback-no-nx)")

    import networkx as nx  # 可选后端

    last_error: Exception | None = None
    for attempt in range(12):
        try:
            g = nx.LFR_benchmark_graph(
                n=n,
                tau1=tau1,
                tau2=tau2,
                mu=mu,
                average_degree=avg_degree,
                max_degree=max_degree,
                min_community=min_community,
                seed=int(seed) + attempt,
            )
            adj = nx.to_numpy_array(g, dtype=np.float64)
            np.fill_diagonal(adj, 0.0)
            adj = (adj > 0).astype(np.float64)
            adj = _largest_component(adj)
            if adj.shape[0] < max(20, int(0.5 * n)):
                raise GraphBuildError("LFR 最大连通分量过小", code="E201")
            return GraphData(adjacency=adj, name="lfr")
        except Exception as exc:  # noqa: BLE001 - LFR 抽样常见失败
            last_error = exc
    return degree_corrected_sbm(
        n=n, rng=rng, name=f"dc-sbm(lfr-fallback:{type(last_error).__name__})"
    )


def karate_club() -> GraphData:
    """内置真实小图：Zachary 空手道俱乐部（34 节点 / 78 边）。

    networkx 可用时以 networkx 权威数据为准；否则使用内置硬编码边表。
    """
    if available_networkx():
        try:
            import networkx as nx

            adj = nx.to_numpy_array(nx.karate_club_graph(), dtype=np.float64)
            np.fill_diagonal(adj, 0.0)
            return GraphData(adjacency=(adj > 0).astype(np.float64), name="karate")
        except Exception:  # noqa: BLE001 - 后端异常时走内置边表
            pass
    edges = [
        (1, 2), (1, 3), (2, 3), (1, 4), (2, 4), (3, 4), (1, 5), (1, 6), (1, 7),
        (5, 7), (6, 7), (1, 8), (2, 8), (3, 8), (4, 8), (1, 9), (3, 9), (3, 10),
        (1, 11), (5, 11), (6, 11), (1, 12), (1, 13), (4, 13), (1, 14), (2, 14),
        (3, 14), (4, 14), (6, 17), (7, 17), (1, 18), (2, 18), (1, 20), (2, 20),
        (1, 22), (2, 22), (24, 26), (25, 26), (3, 28), (24, 28), (25, 28), (3, 29),
        (24, 30), (27, 30), (2, 31), (9, 31), (1, 32), (25, 32), (26, 32), (29, 32),
        (3, 33), (9, 33), (15, 33), (16, 33), (19, 33), (30, 33), (34, 33), (9, 34),
        (10, 34), (14, 34), (15, 34), (16, 34), (19, 34), (20, 34), (21, 34),
        (23, 34), (24, 34), (27, 34), (28, 34), (29, 34), (30, 34), (31, 34),
        (32, 34), (33, 34), (23, 34), (24, 26), (25, 26), (26, 32), (27, 30),
        (5, 7), (6, 11), (1, 5), (1, 6), (1, 7), (2, 31), (9, 31), (14, 34),
    ]
    n = 34
    adj = np.zeros((n, n), dtype=np.float64)
    for u, v in edges:
        a, b = int(u) - 1, int(v) - 1
        if a == b:
            continue
        adj[a, b] = 1.0
        adj[b, a] = 1.0
    np.fill_diagonal(adj, 0.0)
    return GraphData(adjacency=adj, name="karate")
