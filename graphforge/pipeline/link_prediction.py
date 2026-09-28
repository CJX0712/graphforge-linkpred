"""端到端链接预测流水线。

流程：构图 -> 划分（test/valid 边物理删除）-> [可选 HPO，仅用 train/valid]
      -> node2vec 游走与嵌入 -> 边特征 -> 分类器 -> test holdout 评测 -> 误例分析。

作者: 晨星
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from ..core.capabilities import backend_tier, capability_report
from ..core.config import GraphForgeConfig
from ..core.errors import PipelineError
from ..core.seed import derive_seed, make_rng, normalize_seed, set_global_seed
from ..core.types import EdgeSplit, FailureCase, GraphData, SeedResult
from ..data.loader import build_graph
from ..data.split import split_edges
from ..eval.report import analyze_failures, evaluate_scores
from ..hpo.search import SearchResult, tune_hyperparameters
from ..training.trainer import Node2VecLinkPredictor


@dataclass(eq=False)
class PipelineOutput:
    """单次流水线产物。"""

    seed: int
    graph: GraphData
    split: EdgeSplit
    result: SeedResult
    failures: list[FailureCase]
    hpo: SearchResult | None = None
    params: dict[str, object] = field(default_factory=dict)
    meta: dict[str, object] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return {
            "seed": self.seed,
            "graph": {
                "name": self.graph.name,
                "n_nodes": self.graph.n_nodes,
                "n_edges": self.graph.n_edges,
            },
            "split": self.split.meta,
            "metrics": {
                "auc": round(self.result.auc, 6),
                "ap": round(self.result.ap, 6),
                "accuracy": round(self.result.accuracy, 6),
                "seconds": round(self.result.seconds, 3),
            },
            "params": self.params,
            "hpo": self.hpo.as_dict() if self.hpo else None,
            "failures": [f.as_dict() for f in self.failures],
            "meta": self.meta,
        }


class LinkPredictionPipeline:
    """单 seed / 单图配置的端到端流水线。"""

    def __init__(
        self,
        config: GraphForgeConfig,
        seed: int | None = None,
        force_tier1: bool = False,
        tuned_params: dict[str, object] | None = None,
        avg_degree: float | None = None,
    ) -> None:
        self.config = config
        self.seed = normalize_seed(config.seed if seed is None else seed)
        self.force_tier1 = bool(force_tier1)
        self.tuned_params = dict(tuned_params or {})
        self.avg_degree = float(config.avg_degree if avg_degree is None else avg_degree)
        self.graph: GraphData | None = None
        self.split: EdgeSplit | None = None

    # ------------------------------------------------------------------ 数据
    def prepare(self) -> tuple[GraphData, EdgeSplit]:
        set_global_seed(self.seed)
        rng = make_rng(derive_seed(self.seed, "graph"))
        graph = build_graph(
            self.config.graph,
            self.config.n_nodes,
            rng,
            avg_degree=self.avg_degree,
            n_communities=self.config.n_communities,
            homophily=self.config.homophily,
        )
        split_rng = make_rng(derive_seed(self.seed, "split"))
        split = split_edges(
            graph,
            train_ratio=self.config.train_ratio,
            valid_ratio=self.config.valid_ratio,
            test_ratio=self.config.test_ratio,
            negative_ratio=self.config.negative_ratio,
            rng=split_rng,
        )
        self.graph, self.split = graph, split
        return graph, split

    # ------------------------------------------------------------------- HPO
    def tune(self, split: EdgeSplit) -> SearchResult:
        return tune_hyperparameters(
            split,
            self.config,
            seed=self.seed,
            n_trials=self.config.hpo_trials,
            force_tier1=self.force_tier1,
        )

    # -------------------------------------------------------------------- run
    def run(self) -> PipelineOutput:
        started = time.perf_counter()
        graph, split = self.prepare()
        if split.test_edges.shape[0] == 0:
            raise PipelineError("test holdout 为空", code="E502")

        hpo: SearchResult | None = None
        params = dict(self.tuned_params)
        if self.config.hpo_enabled and not params:
            hpo = self.tune(split)
            params = {k: v for k, v in hpo.best_params.items()}

        model = Node2VecLinkPredictor(
            self.config, seed=self.seed, force_tier1=self.force_tier1, **params
        )
        model.fit(split)
        scores = model.predict_proba(split.test_edges)
        result = evaluate_scores(
            method="graphforge-node2vec",
            seed=self.seed,
            labels=split.test_labels,
            scores=scores,
            n_train=int(split.train_edges.shape[0]),
            seconds=time.perf_counter() - started,
            params=model.params_used,
        )
        failures = analyze_failures(split, scores)
        return PipelineOutput(
            seed=self.seed,
            graph=graph,
            split=split,
            result=result,
            failures=failures,
            hpo=hpo,
            params=model.params_used,
            meta={
                "tier": backend_tier(self.force_tier1),
                "capabilities": capability_report(self.force_tier1),
                "walk_samples": model.walk_count,
                "train_pairs": model.pair_count,
                "train_feature_dim": int(np.asarray(model.train_features).shape[1]),
                "total_seconds": round(time.perf_counter() - started, 3),
            },
        )
