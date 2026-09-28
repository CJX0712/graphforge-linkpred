"""data 层单测：图生成与划分的防泄漏不变量。"""

from __future__ import annotations

import numpy as np
import pytest

from graphforge.core.errors import GraphBuildError, SplitError
from graphforge.core.seed import derive_seed, make_rng
from graphforge.core.types import GraphData
from graphforge.data.loader import build_graph, from_edge_list, load_builtin
from graphforge.data.split import split_edges
from graphforge.data.synthetic import (
    degree_corrected_sbm,
    karate_club,
    lfr_graph,
    stochastic_block_model,
)


def _is_connected(adj: np.ndarray) -> bool:
    n = adj.shape[0]
    seen = np.zeros(n, dtype=bool)
    stack = [0]
    seen[0] = True
    while stack:
        node = stack.pop()
        for nb in np.nonzero(adj[node] > 0)[0]:
            if not seen[nb]:
                seen[nb] = True
                stack.append(int(nb))
    return bool(seen.all())


def test_karate_is_builtin_real_graph():
    g = karate_club()
    assert g.n_nodes == 34
    assert g.n_edges == 78
    assert load_builtin("karate").n_nodes == 34


def test_sbm_generation_properties():
    g = stochastic_block_model(200, n_communities=4, p_in=0.2, p_out=0.01, rng=make_rng(0))
    assert g.n_nodes > 100
    assert np.array_equal(g.adjacency, g.adjacency.T)
    assert np.all(np.diag(g.adjacency) == 0)
    assert _is_connected(g.adjacency)


def test_degree_corrected_sbm_has_heterogeneous_degrees():
    g = degree_corrected_sbm(200, rng=make_rng(1))
    deg = g.degrees
    assert deg.max() > 3 * deg.mean() / 2
    assert _is_connected(g.adjacency)


def test_lfr_generation_or_fallback():
    g = lfr_graph(200, mu=0.3, avg_degree=8.0, seed=3)
    assert g.n_nodes > 20
    assert _is_connected(g.adjacency)


def test_from_edge_list_symmetrizes():
    g = from_edge_list([(0, 1), (1, 2)], n_nodes=3)
    assert g.adjacency[0, 1] == 1.0 and g.adjacency[1, 0] == 1.0
    assert g.n_edges == 2


def test_from_edge_list_rejects_bad_input():
    with pytest.raises(GraphBuildError):
        from_edge_list(np.zeros((0, 2), dtype=np.int64))
    with pytest.raises(GraphBuildError):
        from_edge_list([(0, 5)], n_nodes=3)


def test_build_graph_dispatch():
    for name in ("sbm", "lfr", "karate"):
        g = build_graph(name, 120, make_rng(0))
        assert isinstance(g, GraphData)
    with pytest.raises(GraphBuildError):
        build_graph("unknown", 120, make_rng(0))


def test_split_physically_removes_holdout_edges(karate_split):
    train_adj = karate_split.train_graph.adjacency
    for u, v in karate_split.test_edges[karate_split.test_labels == 1]:
        assert train_adj[u, v] == 0.0
    for u, v in karate_split.valid_edges[karate_split.valid_labels == 1]:
        assert train_adj[u, v] == 0.0


def test_train_graph_stays_connected(karate_split):
    assert _is_connected(karate_split.train_graph.adjacency)


def test_negatives_never_hit_true_edges(karate_split):
    positive = {
        (int(u), int(v)) for u, v in karate_split.positive_edges
    }
    for edges, labels in (
        (karate_split.train_edges, karate_split.train_labels),
        (karate_split.valid_edges, karate_split.valid_labels),
        (karate_split.test_edges, karate_split.test_labels),
    ):
        for (u, v), label in zip(edges, labels):
            if label == 0:
                assert (min(u, v), max(u, v)) not in positive
                assert u != v


def test_split_covers_all_original_edges(karate_split, karate_graph):
    n_train_pos = int(karate_split.train_labels.sum())
    n_valid_pos = int(karate_split.valid_labels.sum())
    n_test_pos = int(karate_split.test_labels.sum())
    assert n_train_pos + n_valid_pos + n_test_pos == karate_graph.n_edges
    assert n_test_pos > 0 and n_valid_pos > 0


def test_split_is_balanced_and_shuffled(karate_split):
    for labels in (
        karate_split.train_labels,
        karate_split.valid_labels,
        karate_split.test_labels,
    ):
        assert labels.sum() == (labels == 0).sum()


def test_split_is_reproducible(karate_graph):
    a = split_edges(karate_graph, 0.7, 0.1, 0.2, 1.0, make_rng(derive_seed(9, "split")))
    b = split_edges(karate_graph, 0.7, 0.1, 0.2, 1.0, make_rng(derive_seed(9, "split")))
    assert np.array_equal(a.test_edges, b.test_edges)
    assert np.array_equal(a.test_labels, b.test_labels)


def test_split_rejects_tiny_graph():
    tiny = GraphData(adjacency=np.array([[0.0, 1.0], [1.0, 0.0]]), name="tiny")
    with pytest.raises(SplitError):
        split_edges(tiny, 0.7, 0.1, 0.2, 1.0, make_rng(0))
