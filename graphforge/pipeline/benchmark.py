"""跨 seed 基准对比：本系统 vs 启发式基线 vs 离线兜底 vs 消融。

统计口径：每个方法跑 >= 3 个 seed，报告 mean ± std；
「胜基线」判定 = 均值更高 且 均值差 > 0.5 * (std1 + std2)。

作者: 晨星
"""

from __future__ import annotations

import time

import numpy as np

from ..core.capabilities import backend_tier, capability_report, forced_tier1
from ..core.config import GraphForgeConfig
from ..core.errors import BenchmarkError
from ..core.seed import derive_seed, make_rng, set_global_seed
from ..core.types import BenchmarkResult, SeedResult
from ..data.loader import build_graph
from ..data.split import split_edges
from ..eval.report import evaluate_scores, format_table, summarize
from ..graph.heuristics import score_edges
from ..hpo.search import SearchResult, tune_hyperparameters
from ..preprocess.features import FeatureAssembler
from ..training.classifier import make_classifier
from ..training.trainer import Node2VecLinkPredictor

HEURISTIC_METHODS = (
    "common_neighbors",
    "adamic_adar",
    "resource_allocation",
    "jaccard",
    "preferential_attachment",
    "katz",
    "spectral",
    "random",
)

MAIN_METHOD = "graphforge-node2vec"
AUGMENTED_METHOD = "graphforge-node2vec+heur"
TIER1_METHOD = "graphforge-node2vec(tier1)"
ABLATION_METHOD = "deepwalk(p=q=1)"


def significance_check(challenger: BenchmarkResult, baseline: BenchmarkResult) -> dict:
    """判定 challenger 是否显著优于 baseline。"""
    mean_diff = challenger.auc_mean - baseline.auc_mean
    threshold = 0.5 * (challenger.auc_std + baseline.auc_std)
    return {
        "challenger": challenger.method,
        "baseline": baseline.method,
        "auc_mean_diff": round(float(mean_diff), 6),
        "std_gate": round(float(threshold), 6),
        "abs_gain": round(float(abs(mean_diff)), 6),
        "beats_by_mean": bool(challenger.auc_mean > baseline.auc_mean),
        "passes_std_gate": bool(mean_diff > threshold),
        "passes_abs_gate": bool(mean_diff >= 0.03),
        "significant": bool(
            mean_diff > threshold and mean_diff >= 0.03
        ),
    }


def _prepare(config: GraphForgeConfig, seed: int, avg_degree: float | None = None):
    set_global_seed(seed)
    graph_rng = make_rng(derive_seed(seed, "graph"))
    graph = build_graph(
        config.graph,
        config.n_nodes,
        graph_rng,
        avg_degree=float(config.avg_degree if avg_degree is None else avg_degree),
        n_communities=config.n_communities,
        homophily=config.homophily,
    )
    split_rng = make_rng(derive_seed(seed, "split"))
    split = split_edges(
        graph,
        train_ratio=config.train_ratio,
        valid_ratio=config.valid_ratio,
        test_ratio=config.test_ratio,
        negative_ratio=config.negative_ratio,
        rng=split_rng,
    )
    return graph, split


def run_benchmark(
    config: GraphForgeConfig,
    seeds: tuple[int, ...] = (0, 1, 2),
    hpo_trials: int | None = None,
    force_tier1: bool = False,
    include_heuristics: bool = True,
    include_augmented: bool = True,
    collect_failures: bool = True,
    avg_degree: float | None = None,
    verbose: bool = True,
) -> dict:
    """跑完整基准对比，返回结构化结果（所有数字均来自真实运行）。"""
    if len(seeds) < 3:
        raise BenchmarkError("性能对比必须 >= 3 个 seed", code="E503", n_seeds=len(seeds))

    started = time.perf_counter()
    per_method: dict[str, list[SeedResult]] = {}
    datasets: list[dict] = []
    failures: list[dict] = []
    hpo_result: SearchResult | None = None
    tuned_params: dict[str, object] = {}

    def _add(method: str, result: SeedResult) -> None:
        per_method.setdefault(method, []).append(result)

    for seed in seeds:
        graph, split = _prepare(config, int(seed), avg_degree)
        if split.test_edges.shape[0] == 0:
            raise BenchmarkError("test holdout 为空", code="E503", seed=seed)
        datasets.append(
            {
                "seed": int(seed),
                "graph": graph.name,
                "n_nodes": graph.n_nodes,
                "n_edges": graph.n_edges,
                "split": dict(split.meta),
            }
        )

        # ---- HPO：只在第一个 seed 的 train/valid 上做一次 ----
        if config.hpo_enabled and hpo_result is None:
            hpo_result = tune_hyperparameters(
                split,
                config,
                seed=int(seed),
                n_trials=hpo_trials,
                force_tier1=force_tier1,
            )
            tuned_params = dict(hpo_result.best_params)

        # ---- 主方法：node2vec（HPO 偏置参数） ----
        main_model = Node2VecLinkPredictor(
            config, seed=int(seed), force_tier1=force_tier1, **tuned_params
        )
        main_model.fit(split)
        main_scores = main_model.predict_proba(split.test_edges)
        _add(
            MAIN_METHOD,
            evaluate_scores(
                MAIN_METHOD,
                int(seed),
                split.test_labels,
                main_scores,
                split.train_edges.shape[0],
                main_model.seconds,
                dict(main_model.params_used),
            ),
        )
        if collect_failures:
            from ..eval.report import analyze_failures

            for case in analyze_failures(split, main_scores, top_k=2):
                payload = case.as_dict()
                payload["seed"] = int(seed)
                payload["method"] = MAIN_METHOD
                failures.append(payload)

        # ---- 离线兜底：同一嵌入 + 纯 numpy LR + 纯 numpy 指标 ----
        with forced_tier1():
            fallback_clf = make_classifier(
                prefer="numpy", max_iter=config.max_iter, force_tier1=True
            ).fit(main_model.train_features, split.train_labels)
            fallback_scores = fallback_clf.predict_proba(
                main_model.features_for(split.test_edges)
            )
            _add(
                TIER1_METHOD,
                evaluate_scores(
                    TIER1_METHOD,
                    int(seed),
                    split.test_labels,
                    fallback_scores,
                    split.train_edges.shape[0],
                    main_model.seconds,
                    {"classifier": fallback_clf.backend()},
                ),
            )

        # ---- 结构特征增强变体（同一嵌入） ----
        if include_augmented:
            assembler = FeatureAssembler(
                operator=config.edge_operator, use_heuristic_features=True, standardize=True
            ).fit(split.train_graph)
            x_train = assembler.fit_transform_train(
                main_model.embedding.matrix, split.train_edges
            )
            clf = make_classifier(
                prefer=config.classifier, max_iter=config.max_iter, force_tier1=force_tier1
            ).fit(x_train, split.train_labels)
            aug_scores = clf.predict_proba(
                assembler.transform(main_model.embedding.matrix, split.test_edges)
            )
            _add(
                AUGMENTED_METHOD,
                evaluate_scores(
                    AUGMENTED_METHOD,
                    int(seed),
                    split.test_labels,
                    aug_scores,
                    split.train_edges.shape[0],
                    main_model.seconds,
                    {"classifier": clf.backend()},
                ),
            )

        # ---- 消融：DeepWalk（p = q = 1，退化为均匀游走） ----
        ablation_params = {k: v for k, v in tuned_params.items() if k not in ("p", "q")}
        ablation_params.update({"p": 1.0, "q": 1.0})
        ablation = Node2VecLinkPredictor(
            config, seed=int(seed), force_tier1=force_tier1, **ablation_params
        )
        ablation.fit(split)
        ablation_scores = ablation.predict_proba(split.test_edges)
        _add(
            ABLATION_METHOD,
            evaluate_scores(
                ABLATION_METHOD,
                int(seed),
                split.test_labels,
                ablation_scores,
                split.train_edges.shape[0],
                ablation.seconds,
                dict(ablation.params_used),
            ),
        )

        # ---- 启发式基线 ----
        if include_heuristics:
            for name in HEURISTIC_METHODS:
                t0 = time.perf_counter()
                scores = score_edges(name, split.train_graph, split.test_edges)
                _add(
                    name,
                    evaluate_scores(
                        name,
                        int(seed),
                        split.test_labels,
                        scores,
                        split.train_edges.shape[0],
                        time.perf_counter() - t0,
                        {"family": "heuristic"},
                    ),
                )

    results: list[BenchmarkResult] = [
        summarize(method, seeds_list) for method, seeds_list in per_method.items()
    ]
    results.sort(key=lambda r: r.auc_mean, reverse=True)

    by_name = {r.method: r for r in results}
    heuristic_best = max(
        (r for r in results if r.method in HEURISTIC_METHODS),
        key=lambda r: r.auc_mean,
        default=None,
    )
    comparisons = []
    if MAIN_METHOD in by_name:
        for candidate in [heuristic_best] + [
            r for r in results if r.method in (TIER1_METHOD, ABLATION_METHOD)
        ]:
            if candidate is not None and candidate.method != MAIN_METHOD:
                comparisons.append(significance_check(by_name[MAIN_METHOD], candidate))

    table = format_table(results)
    if verbose:
        print(table)

    return {
        "config": {
            "graph": config.graph,
            "n_nodes": config.n_nodes,
            "seed_base": config.seed,
            "seeds": list(int(s) for s in seeds),
            "train_ratio": config.train_ratio,
            "valid_ratio": config.valid_ratio,
            "test_ratio": config.test_ratio,
            "dim": config.dim,
            "walk_length": config.walk_length,
            "num_walks": config.num_walks,
            "window": config.window,
            "negative_samples": config.negative_samples,
            "epochs": config.epochs,
            "edge_operator": config.edge_operator,
            "hpo_enabled": config.hpo_enabled,
            "hpo_trials": hpo_trials if hpo_trials is not None else config.hpo_trials,
            "tier": backend_tier(force_tier1),
        },
        "capabilities": capability_report(force_tier1),
        "datasets": datasets,
        "hpo": hpo_result.as_dict() if hpo_result else None,
        "tuned_params": {
            k: (round(v, 6) if isinstance(v, float) else v) for k, v in tuned_params.items()
        },
        "results": [r.as_row() for r in results],
        "per_seed": {
            r.method: [
                {
                    "seed": s.seed,
                    "auc": round(s.auc, 6),
                    "ap": round(s.ap, 6),
                    "accuracy": round(s.accuracy, 6),
                    "seconds": round(s.seconds, 3),
                }
                for s in r.per_seed
            ]
            for r in results
        },
        "table": table,
        "comparisons": comparisons,
        "best_heuristic": heuristic_best.method if heuristic_best else None,
        "failures": failures,
        "total_seconds": round(time.perf_counter() - started, 3),
    }


def to_jsonable(payload: dict) -> dict:
    """把 numpy 类型统一转成 JSON 可序列化类型。"""

    def _clean(obj):
        if isinstance(obj, dict):
            return {k: _clean(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_clean(v) for v in obj]
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return _clean(obj.tolist())
        return obj

    return _clean(payload)
