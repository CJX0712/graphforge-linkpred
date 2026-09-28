"""评测指标：ROC-AUC / Average Precision / Accuracy。

注意：sklearn 指标一律以 `_sk_` 前缀导入，避免与本地同名函数递归调用。

作者: 晨星
"""

from __future__ import annotations

import numpy as np

from ..core.capabilities import available_sklearn
from ..core.errors import MetricError


def _check_binary(y_true: np.ndarray) -> np.ndarray:
    y = np.asarray(y_true, dtype=np.int64).ravel()
    if y.size == 0:
        raise MetricError("标签为空", code="E501")
    if np.unique(y).size < 2:
        raise MetricError("标签只含单一类别，无法计算 AUC/AP", code="E501")
    return y


def _numpy_roc_auc_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """基于秩的 AUC（含并列平均秩），纯 numpy 兜底实现。"""
    y = _check_binary(y_true)
    scores = np.asarray(y_score, dtype=np.float64).ravel()
    if scores.shape[0] != y.shape[0]:
        raise MetricError("标签与打分长度不一致", code="E501")
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_y = y[order]

    boundary = np.ones(sorted_scores.shape[0], dtype=bool)
    boundary[1:] = sorted_scores[1:] != sorted_scores[:-1]
    starts = np.nonzero(boundary)[0]
    ends = np.append(starts[1:], sorted_scores.shape[0])
    average_ranks = 0.5 * (starts + ends - 1) + 1.0
    ranks = np.repeat(average_ranks, ends - starts)

    n_pos = float(sorted_y.sum())
    n_neg = float(sorted_y.shape[0] - n_pos)
    if n_pos == 0.0 or n_neg == 0.0:
        raise MetricError("正负样本必须同时存在", code="E501")
    return float((ranks[sorted_y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _numpy_average_precision_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Average Precision（按打分降序累加），纯 numpy 兜底实现。"""
    y = _check_binary(y_true)
    scores = np.asarray(y_score, dtype=np.float64).ravel()
    order = np.argsort(-scores, kind="mergesort")
    ranked = y[order]
    cum_pos = np.cumsum(ranked)
    precision = cum_pos / np.arange(1, ranked.shape[0] + 1, dtype=np.float64)
    return float((precision * ranked).sum() / float(ranked.sum()))


def roc_auc_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """ROC-AUC；sklearn 可用时用 sklearn，否则纯 numpy 秩实现。"""
    _check_binary(y_true)  # 先校验，避免 sklearn 单类别时只发 warning 并返回 nan
    if available_sklearn():
        try:
            from sklearn.metrics import roc_auc_score as _sk_roc_auc_score

            return float(_sk_roc_auc_score(np.asarray(y_true).ravel(), np.asarray(y_score).ravel()))
        except Exception:  # noqa: BLE001 - 后端异常时降级
            pass
    return _numpy_roc_auc_score(y_true, y_score)


def average_precision_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Average Precision；sklearn 可用时用 sklearn，否则纯 numpy 实现。"""
    _check_binary(y_true)
    if available_sklearn():
        try:
            from sklearn.metrics import average_precision_score as _sk_average_precision_score

            return float(
                _sk_average_precision_score(
                    np.asarray(y_true).ravel(), np.asarray(y_score).ravel()
                )
            )
        except Exception:  # noqa: BLE001 - 后端异常时降级
            pass
    return _numpy_average_precision_score(y_true, y_score)


def accuracy(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> float:
    """阈值 0.5 下的准确率（纯 numpy，无外部依赖）。"""
    y = np.asarray(y_true, dtype=np.int64).ravel()
    prob = np.asarray(y_prob, dtype=np.float64).ravel()
    if y.shape[0] != prob.shape[0] or y.size == 0:
        raise MetricError("标签与预测长度不一致或为空", code="E501")
    return float(((prob >= threshold).astype(np.int64) == y).mean())
