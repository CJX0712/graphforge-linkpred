"""预处理单测：标准化与防泄漏。"""

from __future__ import annotations

import numpy as np
import pytest

from graphforge.core.errors import TrainingError
from graphforge.graph.walks import Node2VecWalker
from graphforge.preprocess.features import (
    DEFAULT_HEURISTIC_FEATURES,
    FeatureAssembler,
    Standardizer,
)


def test_standardizer_zero_mean_unit_std():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(100, 5)) * 3 + 7
    z = Standardizer().fit_transform(x)
    assert np.allclose(z.mean(axis=0), 0.0, atol=1e-10)
    assert np.allclose(z.std(axis=0), 1.0, atol=1e-10)


def test_standardizer_requires_fit():
    with pytest.raises(TrainingError):
        Standardizer().transform(np.zeros((2, 2)))
    with pytest.raises(TrainingError):
        Standardizer().fit(np.zeros((0, 3)))


def test_standardizer_transform_uses_train_statistics():
    train = np.array([[0.0], [2.0]])
    test = np.array([[1.0], [3.0]])
    scaler = Standardizer()
    z_train = scaler.fit_transform(train)
    z_test = scaler.transform(test)
    assert np.allclose(z_train, [[-1.0], [1.0]])
    assert np.allclose(z_test, [[0.0], [2.0]])


def test_constant_column_is_not_divided_by_zero():
    x = np.array([[1.0, 5.0], [1.0, 6.0]])
    z = Standardizer().fit_transform(x)
    assert np.all(np.isfinite(z))
    assert np.allclose(z[:, 0], 0.0)


def test_assembler_feature_dim():
    asm = FeatureAssembler(operator="hadamard", use_heuristic_features=False)
    assert asm.feature_dim(64) == 64 + 3
    asm_heur = FeatureAssembler(operator="concat", use_heuristic_features=True)
    assert asm_heur.feature_dim(64) == 128 + 3 + len(DEFAULT_HEURISTIC_FEATURES)


def test_assembler_requires_fit():
    asm = FeatureAssembler()
    with pytest.raises(TrainingError):
        asm.transform(np.zeros((4, 8)), np.array([[0, 1]]))


def test_assembler_rejects_unknown_heuristic(karate_graph):
    asm = FeatureAssembler(use_heuristic_features=True, heuristic_names=("nonexistent",))
    with pytest.raises(TrainingError):
        asm.fit(karate_graph)


def test_assembled_features_never_see_holdout_edges(karate_split, karate_graph):
    """训练图特征必须比全图特征弱（holdout 边已被物理删除）。"""
    edges = karate_split.test_edges[karate_split.test_labels == 1]
    train_asm = FeatureAssembler(use_heuristic_features=True).fit(karate_split.train_graph)
    full_asm = FeatureAssembler(use_heuristic_features=True).fit(karate_graph)
    emb = np.eye(karate_graph.n_nodes)
    train_block = train_asm._raw(emb, edges)[:, -len(DEFAULT_HEURISTIC_FEATURES):]
    full_block = full_asm._raw(emb, edges)[:, -len(DEFAULT_HEURISTIC_FEATURES):]
    # 至少一条 holdout 边的共同邻居数在全图下严格更多
    assert np.any(full_block[:, 0] > train_block[:, 0])
    assert np.all(full_block[:, 0] >= train_block[:, 0])


def test_assembler_train_transform_is_reproducible(karate_split):
    walker = Node2VecWalker().fit(karate_split.train_graph)
    rng = np.random.default_rng(0)
    walks = walker.sample(rng, num_walks=2, walk_length=6, p=1.0, q=1.0)
    from graphforge.graph.embed import SkipGramNS

    emb = SkipGramNS(dim=8, window=2, negative=2, epochs=1, batch_size=256, seed=0).fit(
        karate_split.train_graph, walks
    )
    x1 = FeatureAssembler().fit(karate_split.train_graph).fit_transform_train(
        emb.matrix, karate_split.train_edges
    )
    x2 = FeatureAssembler().fit(karate_split.train_graph).fit_transform_train(
        emb.matrix, karate_split.train_edges
    )
    assert np.array_equal(x1, x2)
    assert x1.shape[0] == karate_split.train_edges.shape[0]
