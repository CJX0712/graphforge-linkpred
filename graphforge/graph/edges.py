"""边特征算子：把两端节点嵌入映射为一条边的特征向量。

支持 hadamard / average / weighted_l1 / weighted_l2 / concat 五种经典算子。

作者: 晨星
"""

from __future__ import annotations

import numpy as np

from ..core.errors import EdgeOperatorError


def hadamard(matrix: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """逐元素乘积（node2vec 论文默认算子）。"""
    return matrix[edges[:, 0]] * matrix[edges[:, 1]]


def average(matrix: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """逐元素均值。"""
    return (matrix[edges[:, 0]] + matrix[edges[:, 1]]) * 0.5


def weighted_l1(matrix: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """逐元素差的绝对值。"""
    return np.abs(matrix[edges[:, 0]] - matrix[edges[:, 1]])


def weighted_l2(matrix: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """逐元素差的平方。"""
    diff = matrix[edges[:, 0]] - matrix[edges[:, 1]]
    return diff * diff


def concat(matrix: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """两端嵌入拼接。"""
    return np.concatenate([matrix[edges[:, 0]], matrix[edges[:, 1]]], axis=1)


OPERATORS = {
    "hadamard": hadamard,
    "average": average,
    "weighted_l1": weighted_l1,
    "weighted_l2": weighted_l2,
    "concat": concat,
}


def available_operators() -> tuple[str, ...]:
    return tuple(sorted(OPERATORS))


def get_operator(name: str):
    """按名称取算子函数（大小写不敏感）。"""
    key = str(name).strip().lower()
    if key not in OPERATORS:
        raise EdgeOperatorError(
            f"未知边算子: {name}，可选 {available_operators()}", code="E303", name=name
        )
    return OPERATORS[key]


def apply_edge_operator(matrix: np.ndarray, edges: np.ndarray, name: str) -> np.ndarray:
    """应用边算子，返回 (m, d) 或 (m, 2d) 特征矩阵。"""
    matrix = np.asarray(matrix, dtype=np.float64)
    edges = np.asarray(edges, dtype=np.int64)
    if edges.ndim != 2 or edges.shape[1] != 2:
        raise EdgeOperatorError("edges 形状必须是 (m, 2)", code="E303", shape=edges.shape)
    if edges.size and (edges.max() >= matrix.shape[0] or edges.min() < 0):
        raise EdgeOperatorError("边索引越界", code="E303")
    return get_operator(name)(matrix, edges)
