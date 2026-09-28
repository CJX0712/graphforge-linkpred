"""训练层单测：分类器后端、降级路径、训练器确定性。"""

from __future__ import annotations

import numpy as np
import pytest

from graphforge.core import capabilities as caps
from graphforge.core.config import load_config
from graphforge.core.errors import TrainingError
from graphforge.training.classifier import (
    NumpyLogisticRegression,
    SklearnLogisticRegression,
    make_classifier,
)
from graphforge.training.trainer import Node2VecLinkPredictor


def _separable_data(n: int = 200, seed: int = 0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 3))
    y = (x[:, 0] + 0.5 * x[:, 1] > 0).astype(np.int64)
    return x, y


def test_numpy_logistic_regression_learns_separable_problem():
    x, y = _separable_data()
    model = NumpyLogisticRegression(c=1.0, max_iter=50).fit(x, y)
    prob = model.predict_proba(x)
    assert model.backend() == "numpy"
    assert np.mean((prob >= 0.5).astype(int) == y) > 0.95


def test_numpy_logistic_regression_matches_sklearn_direction():
    x, y = _separable_data(400, seed=1)
    try:
        sk = SklearnLogisticRegression(c=1.0, max_iter=1000).fit(x, y)
    except Exception:  # pragma: no cover - sklearn 缺失
        pytest.skip("sklearn 不可用")
    np_pred = NumpyLogisticRegression(c=1.0, max_iter=50).fit(x, y).predict_proba(x)
    sk_pred = sk.predict_proba(x)
    assert np.mean(np.abs(np_pred - sk_pred)) < 0.05
    assert np.mean((np_pred >= 0.5) == (sk_pred >= 0.5)) > 0.98


def test_classifier_rejects_single_class():
    with pytest.raises(TrainingError):
        NumpyLogisticRegression().fit(np.zeros((10, 2)), np.ones(10))
    with pytest.raises(TrainingError):
        NumpyLogisticRegression().fit(np.zeros((10, 2)), np.ones(11))


def test_classifier_requires_fit():
    with pytest.raises(TrainingError):
        NumpyLogisticRegression().predict_proba(np.zeros((2, 2)))


def test_make_classifier_backend_selection():
    assert make_classifier("numpy").backend() == "numpy"
    assert make_classifier("auto", force_tier1=True).backend() == "numpy"
    with caps.forced_tier1():
        assert make_classifier("auto").backend() == "numpy"


def test_make_classifier_sklearn_when_available():
    has_sklearn = caps._has("sklearn")
    clf = make_classifier("sklearn") if has_sklearn else make_classifier("numpy")
    assert clf.backend() in ("sklearn", "numpy")
    if has_sklearn:
        assert clf.backend() == "sklearn"


def _config():
    return load_config(dim=16, walk_length=8, num_walks=2, window=2, negative_samples=3,
                       epochs=1, batch_size=256, graph="karate", n_nodes=34)


def test_predictor_is_bitwise_reproducible(karate_split):
    cfg = _config()
    a = Node2VecLinkPredictor(cfg, seed=0).fit(karate_split)
    b = Node2VecLinkPredictor(cfg, seed=0).fit(karate_split)
    assert np.array_equal(a.embedding.matrix, b.embedding.matrix)
    assert np.array_equal(
        a.predict_proba(karate_split.test_edges), b.predict_proba(karate_split.test_edges)
    )
    assert a.params_used["dim"] == 16
    assert a.walk_count == 34 * 2 * 8
    assert a.train_features.shape[0] == karate_split.train_edges.shape[0]


def test_predictor_tier1_fallback_runs(karate_split):
    cfg = _config()
    model = Node2VecLinkPredictor(cfg, seed=0, force_tier1=True)
    with caps.forced_tier1():
        model.fit(karate_split)
        scores = model.predict_proba(karate_split.test_edges)
    assert model.classifier.backend() == "numpy"
    assert scores.shape == (karate_split.test_edges.shape[0],)
    assert np.all(np.isfinite(scores)) and np.all((scores >= 0) & (scores <= 1))


def test_predictor_overrides_take_effect(karate_split):
    cfg = _config()
    model = Node2VecLinkPredictor(cfg, seed=0, p=0.5, q=2.0, dim=8, learning_rate=0.02)
    model.fit(karate_split)
    assert model.params_used["p"] == 0.5
    assert model.params_used["q"] == 2.0
    assert model.params_used["dim"] == 8
    assert model.embedding.matrix.shape[1] == 8


def test_predictor_requires_fit():
    with pytest.raises(TrainingError):
        Node2VecLinkPredictor(_config()).predict_proba(np.array([[0, 1]]))
    with pytest.raises(TrainingError):
        Node2VecLinkPredictor(_config()).train_features
