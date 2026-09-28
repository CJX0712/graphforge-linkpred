"""node2vec 游走单测：不变量 + 退化 + 确定性。"""

from __future__ import annotations

import numpy as np
import pytest

from graphforge.core.errors import WalkError
from graphforge.core.seed import make_rng
from graphforge.core.types import GraphData
from graphforge.graph.walks import Node2VecWalker


@pytest.fixture
def walker(triangle_graph):
    return Node2VecWalker().fit(triangle_graph)


def test_unfitted_walker_raises():
    with pytest.raises(WalkError):
        Node2VecWalker().transition_probabilities(np.array([0]), np.array([-1]), 1.0, 1.0)


def test_all_isolated_graph_raises():
    empty = GraphData(adjacency=np.zeros((3, 3)), name="empty")
    with pytest.raises(WalkError):
        Node2VecWalker().fit(empty)


def test_transition_probabilities_sum_to_one(walker):
    cur = np.array([0, 1, 2, 2, 3])
    prev = np.array([-1, -1, 0, 1, 2])
    for p, q in ((0.25, 0.25), (1.0, 1.0), (0.5, 2.0), (4.0, 0.5)):
        probs = walker.transition_probabilities(cur, prev, p, q)
        row_sum = probs.sum(axis=1)
        # 每个 cur 都有邻居 -> 每行求和为 1
        assert np.allclose(row_sum, 1.0), (p, q, row_sum)


def test_p_equals_q_equals_one_degenerates_to_uniform(walker):
    """p = q = 1 时二阶偏置退化为按边权的均匀游走。"""
    cur = np.array([2, 2, 2])
    prev = np.array([-1, 0, 1])
    probs = walker.transition_probabilities(cur, prev, p=1.0, q=1.0)
    # 节点 2 的邻居是 {0, 1, 3}，均匀各 1/3
    expected = np.tile(np.array([1 / 3, 1 / 3, 1 / 3]), (3, 1))
    assert np.allclose(probs, expected)
    first = walker.transition_probabilities(cur[:1], prev[:1], p=1.0, q=1.0, first_step=True)
    assert np.allclose(first, expected[:1])


def test_first_step_ignores_prev(walker):
    cur = np.array([2])
    for prev in (-1, 0, 1, 3):
        probs = walker.transition_probabilities(cur, np.array([prev]), 0.5, 4.0,
                                                first_step=True)
        assert np.allclose(probs, np.array([[1 / 3, 1 / 3, 1 / 3]]))


def test_biased_transition_matches_hand_computation(walker):
    """cur=2, prev=0, 候选 {0,1,3}: alpha = [1/p, 1 (0-1 有边), 1/q]。"""
    probs = walker.transition_probabilities(np.array([2]), np.array([0]), p=1.0, q=2.0)
    expected = np.array([[0.4, 0.4, 0.2]])  # [1, 1, 0.5] 归一化
    assert np.allclose(probs, expected)

    probs_small_q = walker.transition_probabilities(np.array([2]), np.array([0]), p=1.0, q=0.25)
    # q 小 -> 更鼓励外扩到 3
    assert probs_small_q[0, 2] > probs[0, 2]

    probs_small_p = walker.transition_probabilities(np.array([2]), np.array([0]), p=0.25, q=1.0)
    # p 小 -> 更鼓励回退到 0
    assert probs_small_p[0, 0] > probs[0, 0]


def test_large_q_suppresses_outward_moves(walker):
    probs = walker.transition_probabilities(np.array([2]), np.array([0]), p=1.0, q=1e6)
    assert probs[0, 2] < 1e-5


def test_sample_shape_and_validity(walker):
    walks = walker.sample(make_rng(0), num_walks=5, walk_length=12, p=1.0, q=1.0)
    assert walks.shape == (4 * 5, 12)
    adj = walker.adjacency
    starts = walks[:, 0]
    assert np.array_equal(starts, np.tile(np.arange(4), 5))
    for step in range(walks.shape[1] - 1):
        assert np.all(adj[walks[:, step], walks[:, step + 1]] > 0)


def test_sample_is_bitwise_reproducible(walker):
    a = walker.sample(make_rng(42), 4, 16, 0.5, 2.0)
    b = walker.sample(make_rng(42), 4, 16, 0.5, 2.0)
    assert np.array_equal(a, b)
    c = walker.sample(make_rng(43), 4, 16, 0.5, 2.0)
    assert not np.array_equal(a, c)


def test_dead_end_walker_stays_put():
    adjacency = np.zeros((3, 3), dtype=np.float64)
    adjacency[0, 1] = adjacency[1, 0] = 1.0
    graph = GraphData(adjacency=adjacency, name="with-isolated")
    w = Node2VecWalker().fit(graph)
    walks = w.sample(make_rng(0), num_walks=2, walk_length=6, p=1.0, q=1.0)
    # 节点 2 是孤立点，游走必须停在 2
    isolated_rows = np.nonzero(walks[:, 0] == 2)[0]
    assert isolated_rows.size == 2
    assert np.all(walks[isolated_rows] == 2)


def test_sample_rejects_bad_arguments(walker):
    with pytest.raises(WalkError):
        walker.sample(make_rng(0), 1, 10, p=0.0, q=1.0)
    with pytest.raises(WalkError):
        walker.sample(make_rng(0), 0, 10, p=1.0, q=1.0)
    with pytest.raises(WalkError):
        walker.sample(make_rng(0), 1, 1, p=1.0, q=1.0)


def test_transition_weights_shape_mismatch(walker):
    with pytest.raises(WalkError):
        walker.transition_weights(np.array([0, 1]), np.array([1]), 1.0, 1.0)


def test_walks_cover_graph(walker):
    walks = walker.sample(make_rng(7), num_walks=20, walk_length=30, p=1.0, q=2.0)
    assert set(np.unique(walks).tolist()) == {0, 1, 2, 3}
