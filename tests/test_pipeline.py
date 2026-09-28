"""流水线单测：端到端跑通 + 确定性 + 基准表。"""

from __future__ import annotations

import numpy as np
import pytest

from graphforge.core.config import load_config
from graphforge.core.errors import BenchmarkError
from graphforge.pipeline.benchmark import (
    ABLATION_METHOD,
    HEURISTIC_METHODS,
    MAIN_METHOD,
    TIER1_METHOD,
    run_benchmark,
    significance_check,
)
from graphforge.pipeline.link_prediction import LinkPredictionPipeline


def _small_config(**over):
    base = dict(
        graph="karate",
        n_nodes=34,
        dim=8,
        walk_length=8,
        num_walks=2,
        window=2,
        negative_samples=3,
        epochs=1,
        batch_size=128,
        hpo_enabled=False,
    )
    base.update(over)
    return load_config(**base)


def test_pipeline_run_is_deterministic():
    cfg = _small_config(seed=0)
    a = LinkPredictionPipeline(cfg, seed=0).run()
    b = LinkPredictionPipeline(cfg, seed=0).run()
    assert a.result.auc == b.result.auc
    assert np.array_equal(a.graph.adjacency, b.graph.adjacency)
    assert np.array_equal(a.split.test_edges, b.split.test_edges)
    assert 0.0 <= a.result.auc <= 1.0
    assert a.failures, "误例分析必须产出错例"


def test_pipeline_output_serializable():
    cfg = _small_config(seed=1)
    payload = LinkPredictionPipeline(cfg, seed=1).run().as_dict()
    assert payload["graph"]["name"] == "karate"
    assert payload["split"]["n_test_pos"] > 0
    assert 0.0 <= payload["metrics"]["auc"] <= 1.0
    assert payload["meta"]["walk_samples"] > 0
    assert payload["failures"]


def test_pipeline_different_seeds_differ():
    cfg = _small_config(seed=0)
    a = LinkPredictionPipeline(cfg, seed=0).run()
    b = LinkPredictionPipeline(cfg, seed=1).run()
    assert not np.array_equal(a.split.test_edges, b.split.test_edges)


def test_pipeline_hpo_uses_only_train_valid():
    cfg = _small_config(seed=0, hpo_enabled=True, hpo_trials=2)
    out = LinkPredictionPipeline(cfg, seed=0).run()
    assert out.hpo is not None
    assert out.hpo.n_trials == 2
    assert set(out.params) >= {"p", "q", "dim"}


def test_benchmark_runs_all_methods():
    payload = run_benchmark(_small_config(seed=0), seeds=(0, 1, 2), verbose=False)
    methods = {row["method"] for row in payload["results"]}
    assert MAIN_METHOD in methods
    assert TIER1_METHOD in methods
    assert ABLATION_METHOD in methods
    assert set(HEURISTIC_METHODS) <= methods
    assert len(payload["per_seed"][MAIN_METHOD]) == 3
    assert payload["best_heuristic"] in HEURISTIC_METHODS


def test_benchmark_is_reproducible():
    cfg = _small_config(seed=0)
    a = run_benchmark(cfg, seeds=(0, 1, 2), verbose=False)
    b = run_benchmark(cfg, seeds=(0, 1, 2), verbose=False)
    rows_a = {r["method"]: r["auc_mean"] for r in a["results"]}
    rows_b = {r["method"]: r["auc_mean"] for r in b["results"]}
    assert rows_a == rows_b


def test_benchmark_requires_three_seeds():
    with pytest.raises(BenchmarkError):
        run_benchmark(_small_config(seed=0), seeds=(0, 1), verbose=False)


def test_benchmark_tier1_forced_still_runs():
    from graphforge.core import capabilities as caps

    cfg = _small_config(seed=0, tier1_only=True)
    with caps.forced_tier1():
        payload = run_benchmark(cfg, seeds=(0, 1, 2), force_tier1=True, verbose=False)
    assert payload["capabilities"]["tier"] == "tier1"
    assert MAIN_METHOD in {r["method"] for r in payload["results"]}


def test_significance_check_logic():
    from graphforge.core.types import BenchmarkResult

    strong = BenchmarkResult("a", [0, 1, 2], 0.95, 0.005, 0.9, 0.005)
    weak = BenchmarkResult("b", [0, 1, 2], 0.80, 0.005, 0.7, 0.005)
    verdict = significance_check(strong, weak)
    assert verdict["significant"] is True
    assert verdict["passes_std_gate"] is True
    assert verdict["passes_abs_gate"] is True

    close = BenchmarkResult("c", [0, 1, 2], 0.81, 0.005, 0.7, 0.005)
    verdict_close = significance_check(close, weak)
    assert verdict_close["significant"] is False

    # 方差过大：均值差 0.02 < 0.5 * (0.10 + 0.005) = 0.0525 -> 不通过 std 闸门
    noisy = BenchmarkResult("d", [0, 1, 2], 0.82, 0.10, 0.7, 0.10)
    verdict_noisy = significance_check(noisy, weak)
    assert verdict_noisy["passes_std_gate"] is False
    assert verdict_noisy["significant"] is False
