"""超参搜索：Optuna TPE（可用时）/ 确定性网格搜索（兜底）。

搜索只在 train 折上训练、在 valid 折上打分，test 折全程不参与。

作者: 晨星
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..core.capabilities import available_optuna
from ..core.config import GraphForgeConfig
from ..core.errors import HPOError
from ..core.seed import derive_seed, normalize_seed
from ..core.types import EdgeSplit
from ..eval.metrics import roc_auc_score
from ..training.trainer import Node2VecLinkPredictor

SEARCH_SPACE: dict[str, tuple] = {
    "p": (0.25, 4.0),
    "q": (0.25, 4.0),
    "dim": (32, 64, 128),
}

GRID_CANDIDATES: tuple[dict[str, float], ...] = (
    {"p": 0.5, "q": 2.0, "dim": 64},
    {"p": 1.0, "q": 1.0, "dim": 64},
    {"p": 1.0, "q": 0.5, "dim": 64},
    {"p": 2.0, "q": 0.5, "dim": 64},
    {"p": 0.5, "q": 0.5, "dim": 32},
    {"p": 2.0, "q": 2.0, "dim": 32},
    {"p": 0.25, "q": 4.0, "dim": 128},
    {"p": 4.0, "q": 0.25, "dim": 128},
    {"p": 1.0, "q": 2.0, "dim": 128},
    {"p": 0.5, "q": 1.0, "dim": 32},
)


@dataclass(eq=False)
class SearchResult:
    """超参搜索结果。"""

    best_params: dict[str, object]
    best_score: float
    n_trials: int
    backend: str
    history: list[dict[str, object]] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        return {
            "backend": self.backend,
            "n_trials": self.n_trials,
            "best_score": round(float(self.best_score), 6),
            "best_params": {
                k: (round(v, 6) if isinstance(v, float) else v) for k, v in self.best_params.items()
            },
            "history": [
                {
                    "params": {
                        k: (round(v, 6) if isinstance(v, float) else v)
                        for k, v in h["params"].items()  # type: ignore[index]
                    },
                    "score": round(float(h["score"]), 6),  # type: ignore[arg-type]
                }
                for h in self.history
            ],
        }


def evaluate_candidate(
    split: EdgeSplit,
    config: GraphForgeConfig,
    params: dict[str, object],
    seed: int,
    force_tier1: bool = False,
) -> float:
    """训练一个候选配置，返回 valid 折 AUC。"""
    if split.valid_edges.size == 0:
        raise HPOError("valid 折为空，无法做超参搜索", code="E401")
    model = Node2VecLinkPredictor(
        config, seed=seed, force_tier1=force_tier1, **params
    )
    model.fit(split)
    scores = model.predict_proba(split.valid_edges)
    return float(roc_auc_score(split.valid_labels, scores))


def grid_search(
    split: EdgeSplit,
    config: GraphForgeConfig,
    seed: int | None = None,
    n_trials: int = 8,
    force_tier1: bool = False,
) -> SearchResult:
    """确定性网格搜索（Optuna 缺失时的 Tier-1 兜底）。"""
    base_seed = normalize_seed(config.seed if seed is None else seed)
    history: list[dict[str, object]] = []
    best_score = -np.inf
    best_params: dict[str, object] = {}
    trials = min(int(n_trials), len(GRID_CANDIDATES))
    for index, candidate in enumerate(GRID_CANDIDATES[:trials]):
        trial_seed = derive_seed(base_seed, f"grid-{index}")
        score = evaluate_candidate(split, config, dict(candidate), trial_seed, force_tier1)
        history.append({"params": dict(candidate), "score": score})
        if score > best_score:
            best_score, best_params = score, dict(candidate)
    if not history:
        raise HPOError("网格搜索未产生任何试验", code="E401")
    return SearchResult(
        best_params=best_params, best_score=best_score, n_trials=len(history),
        backend="grid", history=history,
    )


def tune_hyperparameters(
    split: EdgeSplit,
    config: GraphForgeConfig,
    seed: int | None = None,
    n_trials: int | None = None,
    force_tier1: bool = False,
) -> SearchResult:
    """超参搜索主入口。Optuna 可用且未被强制 Tier-1 时用 TPE，否则网格搜索。"""
    base_seed = normalize_seed(config.seed if seed is None else seed)
    trials = int(n_trials if n_trials is not None else config.hpo_trials)
    if trials < 1:
        raise HPOError("hpo_trials 必须 >= 1", code="E401", trials=trials)

    if force_tier1 or not available_optuna():
        return grid_search(split, config, base_seed, trials, force_tier1)

    import optuna  # 可选后端

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    history: list[dict[str, object]] = []

    def objective(trial: "optuna.Trial") -> float:
        params = {
            "p": float(trial.suggest_float("p", 0.25, 4.0, log=True)),
            "q": float(trial.suggest_float("q", 0.25, 4.0, log=True)),
            "dim": int(trial.suggest_categorical("dim", (32, 64, 128))),
        }
        trial_seed = derive_seed(base_seed, f"optuna-{trial.number}")
        score = evaluate_candidate(split, config, params, trial_seed, force_tier1)
        history.append({"params": dict(params), "score": score})
        return score

    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=base_seed)
    )
    study.optimize(objective, n_trials=trials, catch=())

    best = {k: v for k, v in study.best_params.items()}
    return SearchResult(
        best_params=best,
        best_score=float(study.best_value),
        n_trials=len(history),
        backend="optuna",
        history=history,
    )
