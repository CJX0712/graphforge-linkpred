"""图载入与构造入口。

作者: 晨星
"""

from __future__ import annotations

import numpy as np

from ..core.capabilities import available_networkx
from ..core.errors import GraphBuildError
from ..core.types import GraphData
from .synthetic import karate_club, lfr_graph, sbm_from_avg_degree

BUILTIN_GRAPHS = ("lfr", "sbm", "karate")


def from_edge_list(
    edges: np.ndarray | list[tuple[int, int]],
    n_nodes: int | None = None,
    weights: np.ndarray | None = None,
    name: str = "edge_list",
) -> GraphData:
    """从边表构造无向图（自动对称化、去自环）。"""
    arr = np.asarray(edges, dtype=np.int64)
    if arr.size == 0:
        raise GraphBuildError("边表为空", code="E201")
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise GraphBuildError("边表形状必须是 (m, 2)", code="E201", shape=arr.shape)
    if n_nodes is None:
        n_nodes = int(arr.max()) + 1
    if arr.min() < 0 or arr.max() >= n_nodes:
        raise GraphBuildError("边表节点编号越界", code="E201", n_nodes=n_nodes)
    w = np.ones(arr.shape[0], dtype=np.float64) if weights is None else np.asarray(weights, float)
    if w.shape[0] != arr.shape[0]:
        raise GraphBuildError("weights 长度与边表不一致", code="E201")
    adj = np.zeros((n_nodes, n_nodes), dtype=np.float64)
    adj[arr[:, 0], arr[:, 1]] = w
    adj[arr[:, 1], arr[:, 0]] = w
    np.fill_diagonal(adj, 0.0)
    return GraphData(adjacency=adj, name=name)


def from_networkx(graph: object, name: str = "networkx") -> GraphData:
    """从 networkx.Graph 转换；networkx 不可用时抛错由上层降级。"""
    if not available_networkx():
        raise GraphBuildError("networkx 不可用，无法从 networkx 图转换", code="E201")
    import networkx as nx  # 可选后端

    if not isinstance(graph, nx.Graph):
        raise GraphBuildError("输入不是 networkx.Graph", code="E201")
    adj = nx.to_numpy_array(graph, dtype=np.float64)
    np.fill_diagonal(adj, 0.0)
    return GraphData(adjacency=(adj > 0).astype(np.float64), name=name)


def load_builtin(name: str) -> GraphData:
    """载入内置真实小图。"""
    if name == "karate":
        return karate_club()
    raise GraphBuildError(f"未知内置图: {name}", code="E201", name=name)


def build_graph(
    graph_name: str,
    n_nodes: int,
    rng: np.random.Generator,
    avg_degree: float = 10.0,
    mu: float = 0.35,
    n_communities: int = 4,
    homophily: float = 10.0,
) -> GraphData:
    """按名称构造基准图。

    sbm：按目标平均度 avg_degree 与同配比 homophily = p_in / p_out 反解概率；
    lfr：LFR 基准（幂律度分布 + 社群，networkx 抽样失败时降级度修正 SBM）；
    karate：内置真实小图。
    """
    if graph_name == "karate":
        return karate_club()
    if graph_name == "sbm":
        return sbm_from_avg_degree(
            n=n_nodes,
            n_communities=n_communities,
            avg_degree=avg_degree,
            homophily=homophily,
            rng=rng,
            name="sbm",
        )
    if graph_name == "lfr":
        return lfr_graph(n=n_nodes, mu=mu, avg_degree=avg_degree, seed=int(rng.integers(1 << 30)))
    raise GraphBuildError(f"未知图类型: {graph_name}", code="E201", name=graph_name)
