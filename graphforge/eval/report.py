"""报告生成：固定宽度表格 + 跨 seed 聚合 + 误例归因。

作者: 晨星
"""

from __future__ import annotations

import numpy as np

from ..core.types import BenchmarkResult, EdgeSplit, FailureCase, SeedResult
from .metrics import accuracy, average_precision_score, roc_auc_score

COLUMN_WIDTHS = (26, 8, 10, 10, 10, 10)


def _fmt(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def format_table(results: list[BenchmarkResult]) -> str:
    """打印固定宽度基准表（表头用 f\"{h:<10}\" 对齐）。"""
    headers = ("method", "seeds", "auc_mean", "auc_std", "ap_mean", "ap_std")
    widths = COLUMN_WIDTHS
    line_parts = [f"{h:<{w}}" for h, w in zip(headers, widths)]
    lines = ["".join(line_parts), "-" * sum(widths)]
    for res in results:
        row = (
            res.method,
            str(len(res.seeds)),
            _fmt(res.auc_mean),
            _fmt(res.auc_std),
            _fmt(res.ap_mean),
            _fmt(res.ap_std),
        )
        lines.append("".join(f"{value:<{w}}" for value, w in zip(row, widths)))
    return "\n".join(lines)


def summarize(method: str, per_seed: list[SeedResult], notes: str = "") -> BenchmarkResult:
    """把多个 seed 的结果聚合为 mean ± std。"""
    if not per_seed:
        raise ValueError("per_seed 为空，无法聚合")
    aucs = np.array([s.auc for s in per_seed], dtype=np.float64)
    aps = np.array([s.ap for s in per_seed], dtype=np.float64)
    return BenchmarkResult(
        method=method,
        seeds=[s.seed for s in per_seed],
        auc_mean=float(aucs.mean()),
        auc_std=float(aucs.std(ddof=0)),
        ap_mean=float(aps.mean()),
        ap_std=float(aps.std(ddof=0)),
        per_seed=list(per_seed),
        notes=notes,
    )


def evaluate_scores(
    method: str,
    seed: int,
    labels: np.ndarray,
    scores: np.ndarray,
    n_train: int,
    seconds: float,
    params: dict | None = None,
) -> SeedResult:
    """由打分直接计算单 seed 指标。"""
    return SeedResult(
        seed=int(seed),
        method=method,
        auc=roc_auc_score(labels, scores),
        ap=average_precision_score(labels, scores),
        accuracy=accuracy(labels, scores),
        n_train=int(n_train),
        n_test=int(np.asarray(labels).size),
        seconds=float(seconds),
        params=params or {},
    )


def _attribute(
    kind: str, common: int, deg_u: int, deg_v: int, score: float, train_graph
) -> str:
    n = train_graph.n_nodes
    avg_deg = float(train_graph.degrees.mean()) if n else 0.0
    if kind == "false_positive":
        if common >= max(3, int(2 * avg_deg / 3)):
            return (
                f"训练图中该点对共享 {common} 个共同邻居（远高于均值量级），"
                "局部结构高度闭合，但这条边在划分时被移入 holdout 或本就不是真实边；"
                "一阶邻居类方法必然误判。"
            )
        if deg_u * deg_v >= int((2.5 * avg_deg) ** 2):
            return (
                f"两端均为高度节点（deg={deg_u}, deg={deg_v}），度效应使嵌入范数偏大，"
                "Hadamard 内积被放大，模型对 hub-hub 非边过度自信。"
            )
        if common == 0:
            return "两端在训练图中无共同邻居且分属不同社群，模型依据高阶相似误判为存在连边。"
        return "局部与全局证据都偏弱，属于嵌入相似度噪声区域的误报。"
    if common == 0:
        return (
            "该真实边在训练图中被物理删除后，两端失去全部共同邻居，"
            "游走无法在二者间建立共现，嵌入相似度过低 —— 典型的长程/跨社群边漏检。"
        )
    if deg_u <= 1 or deg_v <= 1:
        return (
            f"端点度数极低（deg={deg_u}, deg={deg_v}），游走访问次数不足，"
            "嵌入估计方差大，低度节点边系统性漏检。"
        )
    return (
        f"两端有 {common} 个共同邻居但打分仍偏低，说明该边依赖二阶以上结构，"
        "当前 p/q 偏置下的游走未能充分覆盖其邻域。"
    )


def analyze_failures(
    split: EdgeSplit,
    scores: np.ndarray,
    top_k: int = 3,
) -> list[FailureCase]:
    """挑出最典型的误报 / 漏检各 top_k 条，并给出归因。"""
    labels = np.asarray(split.test_labels, dtype=np.int64)
    edges = np.asarray(split.test_edges, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    train_graph = split.train_graph
    degrees = train_graph.degrees.astype(np.int64)

    positives = np.nonzero(labels == 1)[0]
    negatives = np.nonzero(labels == 0)[0]
    cases: list[FailureCase] = []

    if negatives.size:
        # 误报：负样本中打分最高者
        fp_order = negatives[np.argsort(-scores[negatives])][:top_k]
        for idx in fp_order:
            u, v = int(edges[idx, 0]), int(edges[idx, 1])
            adj = train_graph.adjacency
            common = int(np.count_nonzero(adj[u].astype(bool) & adj[v].astype(bool)))
            cases.append(
                FailureCase(
                    kind="false_positive",
                    u=u,
                    v=v,
                    score=float(scores[idx]),
                    label=0,
                    common_neighbors=common,
                    deg_u=int(degrees[u]),
                    deg_v=int(degrees[v]),
                    attribution=_attribute(
                        "false_positive", common, int(degrees[u]), int(degrees[v]),
                        float(scores[idx]), train_graph,
                    ),
                )
            )
    if positives.size:
        # 漏检：正样本中打分最低者
        fn_order = positives[np.argsort(scores[positives])][:top_k]
        for idx in fn_order:
            u, v = int(edges[idx, 0]), int(edges[idx, 1])
            adj = train_graph.adjacency
            common = int(np.count_nonzero(adj[u].astype(bool) & adj[v].astype(bool)))
            cases.append(
                FailureCase(
                    kind="false_negative",
                    u=u,
                    v=v,
                    score=float(scores[idx]),
                    label=1,
                    common_neighbors=common,
                    deg_u=int(degrees[u]),
                    deg_v=int(degrees[v]),
                    attribution=_attribute(
                        "false_negative", common, int(degrees[u]), int(degrees[v]),
                        float(scores[idx]), train_graph,
                    ),
                )
            )
    return cases
