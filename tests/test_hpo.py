"""HPO 单测：Optuna/网格两条路径 + 只用 train/valid。"""

from __future__ import annotations

import copy

import numpy as np
import pytest

from graphforge.core import capabilities as caps
from graphforge.core.config import load_config
from graphforge.core.errors import HPOError
from graphforge.hpo.search import (
    GRID_CANDIDATES,
    evaluate_candidate,
    grid_search,
    tune_hyperparameters,
)


def _config():
    return load_config(graph="karate", n_nodes=34, dim=8, walk_length=8, num_walks=2,
                       window=2, negative_samples=3, epochs=1, batch_size=256)


def test_grid_search_returns_best(karate_split):
    result = grid_search(karate_split, _config(), seed=0, n_trials=3)
    assert result.backend == "grid"
    assert result.n_trials == 3
    assert set(result.best_params) == {"p", "q", "dim"}
    assert -1.0 <= result.best_score <= 1.0
    assert len(result.history) == 3


def test_grid_search_respects_trial_cap(karate_split):
    result = grid_search(karate_split, _config(), seed=0, n_trials=99)
    assert result.n_trials == len(GRID_CANDIDATES)


def test_search_result_serializable(karate_split):
    payload = grid_search(karate_split, _config(), seed=0, n_trials=2).as_dict()
    assert payload["backend"] == "grid"
    assert isinstance(payload["best_params"]["p"], float)
    assert len(payload["history"]) == 2


def test_tune_falls_back_to_grid_when_tier1(karate_split):
    result = tune_hyperparameters(karate_split, _config(), seed=0, n_trials=2,
                                  force_tier1=True)
    assert result.backend == "grid"
    assert result.n_trials == 2


def test_tune_uses_optuna_when_available(karate_split):
    if not caps._has("optuna"):
        pytest.skip("optuna 不可用")
    result = tune_hyperparameters(karate_split, _config(), seed=0, n_trials=3)
    assert result.backend == "optuna"
    assert result.n_trials == 3
    assert 0.25 <= result.best_params["p"] <= 4.0
    assert 0.25 <= result.best_params["q"] <= 4.0
    assert result.best_params["dim"] in (32, 64, 128)


def test_tune_is_reproducible(karate_split):
    a = tune_hyperparameters(karate_split, _config(), seed=0, n_trials=2, force_tier1=True)
    b = tune_hyperparameters(karate_split, _config(), seed=0, n_trials=2, force_tier1=True)
    assert np.isclose(a.best_score, b.best_score)
    assert a.best_params == b.best_params


def test_evaluate_candidate_requires_valid_split(karate_split):
    score = evaluate_candidate(karate_split, _config(), {"p": 1.0, "q": 1.0}, 0)
    assert 0.0 <= score <= 1.0


def test_hpo_rejects_zero_trials(karate_split):
    with pytest.raises(HPOError):
        tune_hyperparameters(karate_split, _config(), seed=0, n_trials=0)


def test_hpo_never_sees_test_fold(karate_split):
    """不变量：翻转 test 标签后 HPO 结果必须完全不变（目标函数只用 valid 折）。"""
    cfg = _config()
    base = tune_hyperparameters(karate_split, cfg, seed=0, n_trials=4, force_tier1=True)
    flipped = copy.replace(karate_split, test_labels=1 - karate_split.test_labels)
    after = tune_hyperparameters(flipped, cfg, seed=0, n_trials=4, force_tier1=True)
    assert np.isclose(base.best_score, after.best_score), "HPO 不应受 test 标签影响（泄漏）"
    assert base.best_params == after.best_params
