"""评测指标与报告单测。"""

from __future__ import annotations

import numpy as np
import pytest

from graphforge.core.errors import MetricError
from graphforge.eval.metrics import (
    _numpy_average_precision_score,
    _numpy_roc_auc_score,
    accuracy,
    average_precision_score,
    roc_auc_score,
)
from graphforge.eval.report import analyze_failures, evaluate_scores, format_table, summarize


def test_auc_known_value():
    y = np.array([0, 0, 1, 1])
    s = np.array([0.1, 0.4, 0.35, 0.8])
    assert np.isclose(_numpy_roc_auc_score(y, s), 0.75)


def test_auc_perfect_and_inverse():
    y = np.array([0, 0, 1, 1])
    assert np.isclose(_numpy_roc_auc_score(y, np.array([0.1, 0.2, 0.8, 0.9])), 1.0)
    assert np.isclose(_numpy_roc_auc_score(y, np.array([0.8, 0.9, 0.1, 0.2])), 0.0)


def test_auc_ties_average_rank():
    y = np.array([0, 1])
    assert np.isclose(_numpy_roc_auc_score(y, np.array([0.5, 0.5])), 0.5)
    # 并列打分取平均秩: ranks = [1, 2.5, 2.5, 4] -> (2.5 + 4 - 3) / 4 = 0.875
    y2 = np.array([0, 0, 1, 1])
    assert np.isclose(_numpy_roc_auc_score(y2, np.array([0.1, 0.3, 0.3, 0.9])), 0.875)
    assert np.isclose(roc_auc_score(y2, np.array([0.1, 0.3, 0.3, 0.9])), 0.875)


def test_auc_matches_sklearn_when_available():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=300)
    s = rng.random(300)
    ours = _numpy_roc_auc_score(y, s)
    theirs = roc_auc_score(y, s)
    assert np.isclose(ours, theirs, atol=1e-9)


def test_average_precision_known_value():
    y = np.array([0, 1, 0, 1])
    s = np.array([0.1, 0.4, 0.35, 0.8])
    # 降序: y = [1(0.8), 1(0.4), 0(0.35), 0(0.1)]
    # precision@1 = 1, precision@2 = 1 -> AP = (1 + 1)/2 = 1.0
    assert np.isclose(_numpy_average_precision_score(y, s), 1.0)


def test_average_precision_matches_sklearn_when_available():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, size=200)
    s = rng.random(200)
    assert np.isclose(_numpy_average_precision_score(y, s), average_precision_score(y, s),
                     atol=1e-9)


def test_metrics_reject_single_class():
    y = np.array([1, 1, 1])
    with pytest.raises(MetricError):
        roc_auc_score(y, np.array([0.1, 0.2, 0.3]))
    with pytest.raises(MetricError):
        _numpy_average_precision_score(y, np.array([0.1, 0.2, 0.3]))


def test_metrics_reject_length_mismatch():
    with pytest.raises(MetricError):
        _numpy_roc_auc_score(np.array([0, 1]), np.array([0.5]))


def test_accuracy():
    y = np.array([0, 1, 1, 0])
    assert np.isclose(accuracy(y, np.array([0.1, 0.9, 0.6, 0.4])), 1.0)
    assert np.isclose(accuracy(y, np.array([0.9, 0.1, 0.2, 0.8])), 0.0)
    with pytest.raises(MetricError):
        accuracy(np.array([0]), np.array([0.5, 0.6]))


def test_format_table_fixed_width():
    from graphforge.core.types import BenchmarkResult

    rows = [
        BenchmarkResult("alpha", [0, 1, 2], 0.9, 0.01, 0.88, 0.02),
        BenchmarkResult("beta", [0, 1, 2], 0.5, 0.02, 0.4, 0.03),
    ]
    table = format_table(rows)
    lines = table.splitlines()
    width = sum((26, 8, 10, 10, 10, 10))
    assert len(lines) == 4
    # 列名与列宽固定；超长方法名会撑开该行，但表头与分隔线必须等宽
    assert len(lines[0]) == width
    assert len(lines[1]) == width
    assert len(lines[2]) == width
    assert lines[0].startswith("method".ljust(26))
    assert len(lines[1]) == width and set(lines[1]) == {"-"}


def test_summarize_mean_std():
    from graphforge.core.types import SeedResult

    per_seed = [
        SeedResult(0, "m", 0.80, 0.70, 0.7, 10, 10, 1.0),
        SeedResult(1, "m", 0.90, 0.80, 0.7, 10, 10, 1.0),
        SeedResult(2, "m", 1.00, 0.90, 0.7, 10, 10, 1.0),
    ]
    agg = summarize("m", per_seed)
    assert np.isclose(agg.auc_mean, 0.9)
    assert np.isclose(agg.ap_mean, 0.8)
    assert agg.auc_std > 0
    with pytest.raises(ValueError):
        summarize("m", [])


def test_evaluate_scores_and_failure_analysis(karate_split):
    rng = np.random.default_rng(0)
    scores = rng.random(karate_split.test_edges.shape[0])
    result = evaluate_scores("demo", 0, karate_split.test_labels, scores, 10, 0.5)
    assert 0.0 <= result.auc <= 1.0
    cases = analyze_failures(karate_split, scores, top_k=2)
    kinds = {c.kind for c in cases}
    assert "false_positive" in kinds and "false_negative" in kinds
    assert all(c.attribution for c in cases)
    assert len(cases) == 4
