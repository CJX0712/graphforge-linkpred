"""启发式基线单测：对称性 / SVD 最优性 / 边界。"""

from __future__ import annotations

import numpy as np
import pytest

from graphforge.core.errors import HeuristicError
from graphforge.core.seed import make_rng
from graphforge.graph.heuristics import (
    HEURISTICS,
    available_heuristics,
    low_rank_reconstruction,
    random_projection_reconstruction,
    score_edges,
)


def _pairs(n: int, count: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    u = rng.integers(0, n, size=count)
    v = rng.integers(0, n, size=count)
    return np.stack([u, v], axis=1)


@pytest.mark.parametrize(
    "name",
    [
        "common_neighbors",
        "adamic_adar",
        "resource_allocation",
        "jaccard",
        "preferential_attachment",
        "katz",
        "spectral",
        "random",
    ],
)
def test_scorers_are_symmetric(name, karate_graph):
    edges = _pairs(karate_graph.n_nodes, 60, seed=1)
    swapped = edges[:, ::-1].copy()
    forward = score_edges(name, karate_graph, edges)
    backward = score_edges(name, karate_graph, swapped)
    assert forward.shape == (60,)
    assert np.allclose(forward, backward), name


def test_common_neighbors_matches_bruteforce(karate_graph):
    edges = np.array([[0, 1], [0, 2], [0, 33], [5, 6]])
    adj = karate_graph.adjacency > 0
    expected = np.array(
        [int(np.count_nonzero(adj[u].astype(bool) & adj[v].astype(bool))) for u, v in edges],
        dtype=np.float64,
    )
    got = score_edges("common_neighbors", karate_graph, edges)
    assert np.allclose(got, expected)


def test_adamic_adar_matches_bruteforce(karate_graph):
    edges = np.array([[0, 1], [5, 6], [0, 33]])
    adj = karate_graph.adjacency > 0
    deg = karate_graph.degrees
    scores = []
    for u, v in edges:
        common = np.nonzero(adj[u].astype(bool) & adj[v].astype(bool))[0]
        scores.append(float(np.sum(1.0 / np.log(np.maximum(deg[common], 2.0)))))
    assert np.allclose(score_edges("adamic_adar", karate_graph, edges), np.array(scores))


def test_jaccard_bounds(karate_graph):
    edges = _pairs(karate_graph.n_nodes, 200, seed=3)
    scores = score_edges("jaccard", karate_graph, edges)
    assert np.all(scores >= 0.0) and np.all(scores <= 1.0)


def test_preferential_attachment_equals_degree_product(karate_graph):
    edges = np.array([[0, 1], [2, 3]])
    deg = karate_graph.degrees
    expected = deg[edges[:, 0]] * deg[edges[:, 1]]
    assert np.allclose(score_edges("preferential_attachment", karate_graph, edges), expected)


def test_truncated_svd_beats_random_projection(karate_graph):
    """Eckart–Young：秩-k 截断 SVD 的重构误差不高于随机投影基线。"""
    adj = karate_graph.adjacency
    for k in (2, 4, 8, 16):
        _, svd_err = low_rank_reconstruction(adj, k)
        worst_random = max(
            random_projection_reconstruction(adj, k, make_rng(s))[1] for s in range(5)
        )
        assert svd_err <= worst_random + 1e-12, (k, svd_err, worst_random)


def test_low_rank_error_decreases_with_k(karate_graph):
    adj = karate_graph.adjacency
    errs = [low_rank_reconstruction(adj, k)[1] for k in (2, 8, 32)]
    assert errs[0] >= errs[1] >= errs[2]


def test_katz_rejects_invalid_beta(karate_graph):
    with pytest.raises(HeuristicError):
        score_edges("katz", karate_graph, np.array([[0, 1]]), beta=0.99)


def test_unknown_heuristic_raises(karate_graph):
    with pytest.raises(HeuristicError) as exc:
        score_edges("magic", karate_graph, np.array([[0, 1]]))
    assert "E304" in str(exc.value)


def test_scorer_bounds_check(karate_graph):
    scorer = HEURISTICS["common_neighbors"]().fit(karate_graph)
    with pytest.raises(HeuristicError):
        scorer.score(np.array([[0, 999]]))
    with pytest.raises(HeuristicError):
        HEURISTICS["common_neighbors"]().score(np.array([[0, 1]]))


def test_available_heuristics():
    names = available_heuristics()
    assert set(names) == set(HEURISTICS)
    assert "adamic_adar" in names
