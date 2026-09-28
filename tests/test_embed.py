"""SGNS 嵌入单测：数值梯度、负采样分布、样本构造、确定性。"""

from __future__ import annotations

import numpy as np
import pytest

from graphforge.core.errors import EmbeddingError
from graphforge.core.seed import make_rng
from graphforge.graph.embed import (
    SkipGramNS,
    _scatter_add,
    sgns_forward_backward,
)


def _total_loss(vc, ut, un) -> float:
    return float(sgns_forward_backward(vc, ut, un)[0].sum())


def test_numerical_gradient_matches_analytic():
    rng = np.random.default_rng(0)
    b, d, k = 5, 4, 3
    vc = rng.normal(size=(b, d))
    ut = rng.normal(size=(b, d))
    un = rng.normal(size=(b, k, d))

    _, d_vc, d_ut, d_un = sgns_forward_backward(vc, ut, un)
    eps = 1e-6
    for name, tensor, grad in (("vc", vc, d_vc), ("ut", ut, d_ut), ("un", un, d_un)):
        flat_idx = np.ndindex(tensor.shape)
        for idx in flat_idx:
            original = tensor[idx]
            tensor[idx] = original + eps
            plus = _total_loss(vc, ut, un)
            tensor[idx] = original - eps
            minus = _total_loss(vc, ut, un)
            tensor[idx] = original
            numeric = (plus - minus) / (2 * eps)
            assert np.isclose(numeric, grad[idx], atol=1e-5), (name, idx, numeric, grad[idx])


def test_sgns_loss_is_positive_and_finite():
    rng = np.random.default_rng(1)
    vc = rng.normal(size=(8, 6))
    ut = rng.normal(size=(8, 6))
    un = rng.normal(size=(8, 5, 6))
    loss, _, _, _ = sgns_forward_backward(vc, ut, un)
    assert loss.shape == (8,)
    assert np.all(np.isfinite(loss)) and np.all(loss > 0)


def test_sgns_prefers_aligned_positive_pairs():
    """正样本对齐、负样本远离时 loss 必须更低。"""
    d = 8
    a = np.zeros(d)
    a[0] = 1.0
    vc = np.tile(a, (2, 1))
    aligned = np.tile(a, (2, 1))
    un_aligned = np.tile(-a, (2, 5, 1))
    un_opposed = np.tile(a, (2, 5, 1))
    loss_good = sgns_forward_backward(vc, aligned, un_aligned)[0]
    loss_bad = sgns_forward_backward(vc, aligned, un_opposed)[0]
    assert np.all(loss_good < loss_bad)


def test_negative_distribution_is_normalized_and_monotone():
    degrees = np.array([0.0, 1.0, 2.0, 10.0])
    cdf = SkipGramNS.negative_distribution(degrees)
    assert np.isclose(cdf[-1], 1.0)
    assert np.all(np.diff(cdf) >= 0)
    prob = np.diff(np.concatenate([[0.0], cdf]))
    assert np.isclose(prob.sum(), 1.0)
    assert prob[3] > prob[2] > prob[1] > prob[0]
    # degree^{3/4} 单调
    assert np.isclose(prob[2] / prob[1], (2.0 / 1.0) ** 0.75, rtol=1e-9)


def test_negative_distribution_rejects_degenerate():
    with pytest.raises(EmbeddingError):
        SkipGramNS.negative_distribution(np.array([0.0, 0.0]))


def test_build_pairs_window_one():
    walks = np.array([[0, 1, 2, 3]])
    centers, contexts = SkipGramNS.build_pairs(walks, window=1, rng=make_rng(0))
    # 先放 (left, right) 方向，再放 (right, left) 方向
    expected_centers = np.array([0, 1, 2, 1, 2, 3])
    expected_contexts = np.array([1, 2, 3, 0, 1, 2])
    assert np.array_equal(centers, expected_centers)
    assert np.array_equal(contexts, expected_contexts)


def test_build_pairs_never_crosses_walk_boundary():
    walks = np.array([[0, 1, 2], [3, 4, 5]])
    centers, contexts = SkipGramNS.build_pairs(walks, window=5, rng=make_rng(1))
    assert centers.shape == contexts.shape
    # 同一对必然来自同一条游走
    walk_of = np.where(np.isin(centers, [0, 1, 2]), 0, 1)
    walk_ctx = np.where(np.isin(contexts, [0, 1, 2]), 0, 1)
    assert np.array_equal(walk_of, walk_ctx)


def test_build_pairs_rejects_short_walks():
    with pytest.raises(EmbeddingError):
        SkipGramNS.build_pairs(np.array([[0]]), window=3, rng=make_rng(0))


def test_scatter_add_matches_add_at():
    rng = np.random.default_rng(2)
    n, d, rows = 12, 3, 200
    ids = rng.integers(0, n, size=rows)
    grads = rng.normal(size=(rows, d))
    expected = np.zeros((n, d))
    np.add.at(expected, ids, grads)
    got = np.zeros((n, d))
    _scatter_add(got, ids, grads)
    assert np.allclose(got, expected)
    _scatter_add(got, np.zeros(0, dtype=np.int64), np.zeros((0, d)))


def test_fit_is_bitwise_reproducible(karate_split):
    common = dict(dim=16, window=3, negative=3, epochs=1, batch_size=512)
    a = SkipGramNS(seed=5, **common).fit(karate_split.train_graph, _walks(karate_split))
    b = SkipGramNS(seed=5, **common).fit(karate_split.train_graph, _walks(karate_split))
    assert np.array_equal(a.matrix, b.matrix)
    assert a.matrix.shape == (karate_split.train_graph.n_nodes, 16)
    assert np.all(np.isfinite(a.matrix))
    assert len(a.loss_trace) == 1


def test_fit_loss_decreases(karate_split):
    model = SkipGramNS(dim=16, window=3, negative=3, epochs=3, batch_size=512, seed=5)
    result = model.fit(karate_split.train_graph, _walks(karate_split))
    assert len(result.loss_trace) == 3
    assert result.loss_trace[-1] < result.loss_trace[0]


def test_fit_rejects_bad_arguments():
    with pytest.raises(EmbeddingError):
        SkipGramNS(dim=1)
    with pytest.raises(EmbeddingError):
        SkipGramNS(window=0)
    with pytest.raises(EmbeddingError):
        SkipGramNS(negative=0)
    with pytest.raises(EmbeddingError):
        SkipGramNS().fit(_dummy_graph(), np.zeros((0, 4), dtype=np.int64))


def _dummy_graph():
    from graphforge.core.types import GraphData

    adjacency = np.zeros((4, 4))
    adjacency[0, 1] = adjacency[1, 0] = 1.0
    adjacency[1, 2] = adjacency[2, 1] = 1.0
    adjacency[2, 3] = adjacency[3, 2] = 1.0
    return GraphData(adjacency=adjacency, name="chain")


def _walks(split):
    from graphforge.graph.walks import Node2VecWalker

    walker = Node2VecWalker().fit(split.train_graph)
    return walker.sample(make_rng(0), num_walks=3, walk_length=8, p=1.0, q=1.0)
