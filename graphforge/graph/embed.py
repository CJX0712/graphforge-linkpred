"""Skip-Gram with Negative Sampling（SGNS）—— 纯 numpy 手写实现。

参考 word2vec / node2vec：
  * 正样本：游走序列窗口内的 (center, context) 对，窗口长度逐条游走动态采样；
  * 负样本：按 degree^{3/4} 分布采样（word2vec 的 3/4 次幂平滑）；
  * 优化：Adam / SGD，minibatch 向量化，梯度用 reduceat 做 scatter-add。

作者: 晨星
"""

from __future__ import annotations

import numpy as np

from ..core.errors import EmbeddingError
from ..core.seed import make_rng, normalize_seed
from ..core.types import EmbeddingResult, GraphData


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """数值稳定的 sigmoid。"""
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    exp_x = np.exp(x[~pos])
    out[~pos] = exp_x / (1.0 + exp_x)
    return out


def sgns_forward_backward(
    vc: np.ndarray, ut: np.ndarray, un: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """SGNS 前向 + 反向。

    参数
        vc: (B, d) 中心节点输入向量
        ut: (B, d) 正样本输出向量
        un: (B, K, d) 负样本输出向量

    返回
        (loss_sum, d_vc, d_ut, d_un) —— 梯度均相对于「batch 内 loss 之和」。
    """
    # 用 batched matmul 替代 einsum：einsum 在 (B,K,d) 模式上无 BLAS 加速，慢 10x 以上
    pos_score = np.sum(vc * ut, axis=1)                       # (B,)
    neg_score = (un @ vc[:, :, None])[:, :, 0]                # (B, K)
    loss = np.logaddexp(0.0, -pos_score) + np.logaddexp(0.0, neg_score).sum(axis=1)

    g_pos = -_sigmoid(-pos_score)                        # dL/d(pos_score) = -sigma(-s)
    g_neg = _sigmoid(neg_score)                          # dL/d(neg_score) = sigma(s)

    d_vc = g_pos[:, None] * ut + (g_neg[:, None, :] @ un)[:, 0, :]
    d_ut = g_pos[:, None] * vc
    d_un = g_neg[:, :, None] * vc[:, None, :]
    return loss, d_vc, d_ut, d_un


def _scatter_add(target: np.ndarray, ids: np.ndarray, grads: np.ndarray) -> None:
    """把 grads 按 ids 累加到 target 对应行（按行聚合，比 np.add.at 快得多）。"""
    if ids.size == 0:
        return
    order = np.argsort(ids, kind="stable")
    sorted_ids = ids[order]
    sorted_grads = np.ascontiguousarray(grads[order])
    counts = np.bincount(sorted_ids, minlength=target.shape[0])
    present = np.nonzero(counts)[0]
    if present.size == 0:
        return
    offsets = np.zeros(target.shape[0] + 1, dtype=np.int64)
    np.cumsum(counts, out=offsets[1:])
    starts = offsets[present]
    summed = np.add.reduceat(sorted_grads, starts, axis=0)
    target[present] += summed


class SkipGramNS:
    """Skip-Gram + Negative Sampling 节点嵌入器。"""

    def __init__(
        self,
        dim: int = 64,
        window: int = 5,
        negative: int = 5,
        epochs: int = 1,
        learning_rate: float = 0.01,
        batch_size: int = 2048,
        optimizer: str = "adam",
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        seed: int = 0,
    ) -> None:
        if dim < 2:
            raise EmbeddingError("dim 必须 >= 2", code="E302", dim=dim)
        if window < 1:
            raise EmbeddingError("window 必须 >= 1", code="E302", window=window)
        if negative < 1:
            raise EmbeddingError("negative 必须 >= 1", code="E302", negative=negative)
        self.dim = int(dim)
        self.window = int(window)
        self.negative = int(negative)
        self.epochs = int(epochs)
        self.learning_rate = float(learning_rate)
        self.batch_size = int(batch_size)
        self.optimizer = optimizer
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.eps = float(eps)
        self.seed = normalize_seed(seed)
        self.w_in: np.ndarray | None = None
        self.w_out: np.ndarray | None = None

    # ------------------------------------------------------------ 负采样表
    @staticmethod
    def negative_distribution(degrees: np.ndarray) -> np.ndarray:
        """负采样分布 ∝ degree^{3/4}，返回累积分布（末尾为 1）。"""
        deg = np.asarray(degrees, dtype=np.float64)
        powered = np.power(np.maximum(deg, 0.0), 0.75)
        total = powered.sum()
        if total <= 0 or not np.isfinite(total):
            raise EmbeddingError("负采样分布非法", code="E302", total=total)
        prob = powered / total
        cdf = np.cumsum(prob)
        cdf[-1] = 1.0
        return cdf

    def _sample_negatives(
        self, cdf: np.ndarray, rng: np.random.Generator, size: int
    ) -> np.ndarray:
        u = rng.random(size)
        idx = np.searchsorted(cdf, u, side="right")
        return np.clip(idx, 0, cdf.shape[0] - 1).astype(np.int64)

    # ------------------------------------------------------------ 样本构造
    @staticmethod
    def build_pairs(
        walks: np.ndarray, window: int, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray]:
        """从游走序列构造 (center, context) 正样本对，窗口逐条游走动态采样。"""
        seq = np.asarray(walks, dtype=np.int64)
        if seq.ndim != 2:
            raise EmbeddingError("walks 必须是二维 (n_walks, length)", code="E302")
        n_walks, length = seq.shape
        if length < 2:
            raise EmbeddingError("游走长度必须 >= 2", code="E302", length=length)
        dynamic = rng.integers(1, window + 1, size=n_walks)
        centers: list[np.ndarray] = []
        contexts: list[np.ndarray] = []
        for offset in range(1, min(window, length - 1) + 1):
            left = seq[:, :-offset].ravel()
            right = seq[:, offset:].ravel()
            keep = np.repeat(dynamic >= offset, length - offset)
            left = left[keep]
            right = right[keep]
            if left.size == 0:
                continue
            centers.append(left)
            contexts.append(right)
            centers.append(right)
            contexts.append(left)
        if not centers:
            raise EmbeddingError("未能构造任何正样本对", code="E302")
        return np.concatenate(centers), np.concatenate(contexts)

    # ---------------------------------------------------------------- 训练
    def fit(
        self,
        graph: GraphData,
        walks: np.ndarray,
        rng: np.random.Generator | None = None,
    ) -> EmbeddingResult:
        if rng is None:
            rng = make_rng(self.seed)
        n = graph.n_nodes
        dim = self.dim
        if walks.size == 0:
            raise EmbeddingError("游走序列为空", code="E302")

        cdf = self.negative_distribution(graph.degrees)
        centers, contexts = self.build_pairs(walks, self.window, rng)

        w_in = (rng.random((n, dim), dtype=np.float32) - 0.5).astype(np.float32) / dim
        w_out = np.zeros((n, dim), dtype=np.float32)

        m_in = np.zeros_like(w_in)
        v_in = np.zeros_like(w_in)
        m_out = np.zeros_like(w_out)
        v_out = np.zeros_like(w_out)

        n_pairs = centers.shape[0]
        loss_trace: list[float] = []
        step = 0
        base_lr = self.learning_rate

        for epoch in range(self.epochs):
            order = rng.permutation(n_pairs)
            running = 0.0
            seen = 0
            for start in range(0, n_pairs, self.batch_size):
                batch_idx = order[start : start + self.batch_size]
                if batch_idx.size == 0:
                    continue
                c = centers[batch_idx]
                t = contexts[batch_idx]
                batch = c.shape[0]
                neg = self._sample_negatives(cdf, rng, batch * self.negative).reshape(
                    batch, self.negative
                )

                vc = w_in[c]
                ut = w_out[t]
                un = w_out[neg]

                loss, d_vc, d_ut, d_un = sgns_forward_backward(vc, ut, un)
                running += float(loss.sum())
                seen += batch

                grad_in = np.zeros_like(w_in)
                grad_out = np.zeros_like(w_out)
                _scatter_add(grad_in, c, d_vc)
                _scatter_add(grad_out, t, d_ut)
                _scatter_add(grad_out, neg.ravel(), d_un.reshape(batch * self.negative, dim))
                grad_in /= float(batch)
                grad_out /= float(batch)

                if self.optimizer == "adam":
                    step += 1
                    m_in = self.beta1 * m_in + (1.0 - self.beta1) * grad_in
                    v_in = self.beta2 * v_in + (1.0 - self.beta2) * grad_in * grad_in
                    m_out = self.beta1 * m_out + (1.0 - self.beta1) * grad_out
                    v_out = self.beta2 * v_out + (1.0 - self.beta2) * grad_out * grad_out
                    bias1 = 1.0 - self.beta1**step
                    bias2 = 1.0 - self.beta2**step
                    w_in -= (base_lr * (m_in / bias1) / (np.sqrt(v_in / bias2) + self.eps))
                    w_out -= (base_lr * (m_out / bias1) / (np.sqrt(v_out / bias2) + self.eps))
                else:  # sgd + 线性衰减
                    progress = (start + batch) / max(1, n_pairs)
                    lr = base_lr * max(1.0 - progress, 1e-4)
                    w_in -= lr * grad_in
                    w_out -= lr * grad_out
            loss_trace.append(running / max(1, seen))

        if not np.all(np.isfinite(w_in)):
            raise EmbeddingError("嵌入训练发散（出现 NaN/Inf）", code="E302")

        self.w_in = w_in
        self.w_out = w_out
        return EmbeddingResult(
            matrix=w_in.astype(np.float64),
            n_nodes=n,
            dim=dim,
            params={
                "window": self.window,
                "negative": self.negative,
                "epochs": self.epochs,
                "optimizer": self.optimizer,
                "learning_rate": base_lr,
                "batch_size": self.batch_size,
                "n_pairs": int(n_pairs),
            },
            loss_trace=loss_trace,
        )
