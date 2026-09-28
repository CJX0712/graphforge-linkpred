"""边特征算子单测：交换律 / 各算子性质 / 错误处理。"""

from __future__ import annotations

import numpy as np
import pytest

from graphforge.core.errors import EdgeOperatorError
from graphforge.graph.edges import (
    apply_edge_operator,
    available_operators,
    concat,
    get_operator,
    hadamard,
)


@pytest.fixture
def matrix():
    rng = np.random.default_rng(0)
    return rng.normal(size=(10, 6))


def test_hadamard_is_commutative(matrix):
    edges = np.array([[0, 1], [2, 3], [4, 5]])
    swapped = edges[:, ::-1].copy()
    assert np.array_equal(hadamard(matrix, edges), hadamard(matrix, swapped))


def test_all_symmetric_operators_are_commutative(matrix):
    edges = np.array([[0, 1], [2, 3], [4, 5], [6, 7]])
    swapped = edges[:, ::-1].copy()
    for name in ("hadamard", "average", "weighted_l1", "weighted_l2"):
        forward = apply_edge_operator(matrix, edges, name)
        backward = apply_edge_operator(matrix, swapped, name)
        assert np.array_equal(forward, backward), name


def test_operator_output_shapes(matrix):
    edges = np.array([[0, 1], [2, 3]])
    assert apply_edge_operator(matrix, edges, "hadamard").shape == (2, 6)
    assert apply_edge_operator(matrix, edges, "average").shape == (2, 6)
    assert apply_edge_operator(matrix, edges, "weighted_l1").shape == (2, 6)
    assert apply_edge_operator(matrix, edges, "weighted_l2").shape == (2, 6)
    assert apply_edge_operator(matrix, edges, "concat").shape == (2, 12)
    assert np.array_equal(
        concat(matrix, edges), np.concatenate([matrix[[0, 2]], matrix[[1, 3]]], axis=1)
    )


def test_weighted_l1_l2_are_nonnegative(matrix):
    edges = np.array([[0, 1], [2, 3]])
    assert np.all(apply_edge_operator(matrix, edges, "weighted_l1") >= 0)
    assert np.all(apply_edge_operator(matrix, edges, "weighted_l2") >= 0)
    l1 = apply_edge_operator(matrix, edges, "weighted_l1")
    l2 = apply_edge_operator(matrix, edges, "weighted_l2")
    assert np.allclose(l2, l1**2)


def test_average_is_half_sum(matrix):
    edges = np.array([[0, 1]])
    assert np.allclose(apply_edge_operator(matrix, edges, "average"),
                       0.5 * (matrix[0] + matrix[1]))


def test_available_operators_and_lookup():
    names = available_operators()
    assert "hadamard" in names and "concat" in names
    assert get_operator("Hadamard") is hadamard


def test_unknown_operator_raises(matrix):
    with pytest.raises(EdgeOperatorError) as exc:
        get_operator("nope")
    assert "E303" in str(exc.value)


def test_bad_edge_shapes_raise(matrix):
    with pytest.raises(EdgeOperatorError):
        apply_edge_operator(matrix, np.array([[0, 1, 2]]), "hadamard")
    with pytest.raises(EdgeOperatorError):
        apply_edge_operator(matrix, np.array([[0, 99]]), "hadamard")
