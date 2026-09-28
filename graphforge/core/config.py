"""配置：dataclass schema + ENV_GRAPHFORGE_* 覆盖 + 校验。

优先级：显式入参 > 环境变量 > 默认值。

作者: 晨星
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, fields
from typing import Any, get_type_hints

from .errors import ConfigError, ValidationError
from .seed import normalize_seed
from .types import EDGE_OPERATORS

ENV_PREFIX = "GRAPHFORGE_"

_BOOL_TRUE = {"1", "true", "yes", "on"}
_BOOL_FALSE = {"0", "false", "no", "off"}


@dataclass
class GraphForgeConfig:
    """GraphForge 全量配置。"""

    # ---- 随机性 ----
    seed: int = 42

    # ---- 数据 ----
    graph: str = "sbm"  # sbm | lfr | karate
    n_nodes: int = 700
    n_communities: int = 4
    homophily: float = 10.0  # SBM 的 p_in / p_out 同配比
    avg_degree: float = 10.0
    train_ratio: float = 0.7
    valid_ratio: float = 0.1
    test_ratio: float = 0.2
    negative_ratio: float = 1.0

    # ---- node2vec 游走 ----
    p: float = 1.0
    q: float = 1.0
    walk_length: int = 32
    num_walks: int = 6

    # ---- skip-gram ----
    dim: int = 64
    window: int = 5
    negative_samples: int = 5
    epochs: int = 1
    learning_rate: float = 0.01
    batch_size: int = 2048
    optimizer: str = "adam"  # adam | sgd
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8

    # ---- 下游 ----
    edge_operator: str = "hadamard"
    normalize_embeddings: bool = False  # 行 L2 归一化后再做边算子
    use_heuristic_features: bool = False
    classifier: str = "auto"  # auto | sklearn | numpy
    classifier_c: float = 1.0
    max_iter: int = 1000

    # ---- HPO ----
    hpo_enabled: bool = False
    hpo_trials: int = 8

    # ---- 运行 ----
    tier1_only: bool = False
    log_level: str = "INFO"


def _coerce(raw: str, target_type: type) -> Any:
    text = raw.strip()
    if target_type is bool:
        low = text.lower()
        if low in _BOOL_TRUE:
            return True
        if low in _BOOL_FALSE:
            return False
        raise ConfigError(f"布尔字段无法解析: {raw!r}")
    if target_type is int:
        try:
            return int(text)
        except ValueError as exc:
            raise ConfigError(f"整数字段无法解析: {raw!r}") from exc
    if target_type is float:
        try:
            return float(text)
        except ValueError as exc:
            raise ConfigError(f"浮点字段无法解析: {raw!r}") from exc
    return text


def config_from_env(base: GraphForgeConfig | None = None) -> GraphForgeConfig:
    """读取 GRAPHFORGE_* 环境变量覆盖 base 配置。"""
    cfg = base if base is not None else GraphForgeConfig()
    data = asdict(cfg)
    hints = get_type_hints(GraphForgeConfig)
    for f in fields(GraphForgeConfig):
        env_key = ENV_PREFIX + f.name.upper()
        if env_key in os.environ:
            try:
                data[f.name] = _coerce(os.environ[env_key], hints[f.name])
            except ConfigError as exc:
                raise ConfigError(f"{env_key} 解析失败: {exc.message}", code="E102") from exc
    return GraphForgeConfig(**data)


def load_config(**overrides: Any) -> GraphForgeConfig:
    """构造配置：默认值 <- 环境变量 <- 显式入参，随后做 schema 校验。"""
    unknown = set(overrides) - {f.name for f in fields(GraphForgeConfig)}
    if unknown:
        raise ConfigError(f"未知配置项: {sorted(unknown)}", code="E103")
    cfg = config_from_env()
    for key, value in overrides.items():
        if value is not None:
            setattr(cfg, key, value)
    return validate_config(cfg)


def validate_config(cfg: GraphForgeConfig) -> GraphForgeConfig:
    """schema 校验：类型、范围、枚举。失败抛 ValidationError(E101)。"""
    if not isinstance(cfg.seed, int):
        raise ValidationError("seed 必须是 int", code="E101", got=type(cfg.seed).__name__)
    cfg.seed = normalize_seed(cfg.seed)

    if cfg.graph not in ("lfr", "sbm", "karate"):
        raise ValidationError("graph 必须是 lfr / sbm / karate", code="E101", got=cfg.graph)
    if not 20 <= cfg.n_nodes <= 20000:
        raise ValidationError("n_nodes 超出 [20, 20000]", code="E101", got=cfg.n_nodes)
    if cfg.n_communities < 2:
        raise ValidationError("n_communities 必须 >= 2", code="E101", got=cfg.n_communities)
    if cfg.homophily < 1.0:
        raise ValidationError("homophily 必须 >= 1（p_in >= p_out）", code="E101",
                              got=cfg.homophily)
    if cfg.avg_degree <= 0.0:
        raise ValidationError("avg_degree 必须 > 0", code="E101", got=cfg.avg_degree)

    total = cfg.train_ratio + cfg.valid_ratio + cfg.test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValidationError("train/valid/test ratio 之和必须为 1.0", code="E101", got=total)
    for name, value in (
        ("train_ratio", cfg.train_ratio),
        ("valid_ratio", cfg.valid_ratio),
        ("test_ratio", cfg.test_ratio),
    ):
        if value < 0.0:
            raise ValidationError(f"{name} 不能为负", code="E101", got=value)
    if cfg.test_ratio <= 0.0:
        raise ValidationError("test_ratio 必须 > 0（需要独立 holdout）", code="E101")
    if cfg.negative_ratio <= 0.0:
        raise ValidationError("negative_ratio 必须 > 0", code="E101", got=cfg.negative_ratio)

    if cfg.p <= 0.0 or cfg.q <= 0.0:
        raise ValidationError("p / q 必须 > 0", code="E101", p=cfg.p, q=cfg.q)
    if cfg.walk_length < 2:
        raise ValidationError("walk_length 必须 >= 2", code="E101", got=cfg.walk_length)
    if cfg.num_walks < 1:
        raise ValidationError("num_walks 必须 >= 1", code="E101", got=cfg.num_walks)

    if cfg.dim < 2 or cfg.dim > 1024:
        raise ValidationError("dim 超出 [2, 1024]", code="E101", got=cfg.dim)
    if cfg.window < 1:
        raise ValidationError("window 必须 >= 1", code="E101", got=cfg.window)
    if cfg.negative_samples < 1:
        raise ValidationError("negative_samples 必须 >= 1", code="E101")
    if cfg.epochs < 1:
        raise ValidationError("epochs 必须 >= 1", code="E101", got=cfg.epochs)
    if cfg.learning_rate <= 0.0:
        raise ValidationError("learning_rate 必须 > 0", code="E101", got=cfg.learning_rate)
    if cfg.batch_size < 16:
        raise ValidationError("batch_size 必须 >= 16", code="E101", got=cfg.batch_size)
    if cfg.optimizer not in ("adam", "sgd"):
        raise ValidationError("optimizer 必须是 adam / sgd", code="E101", got=cfg.optimizer)
    if not (0.0 < cfg.beta1 < 1.0) or not (0.0 < cfg.beta2 < 1.0):
        raise ValidationError("beta1 / beta2 必须在 (0, 1)", code="E101")

    if cfg.edge_operator not in EDGE_OPERATORS:
        raise ValidationError(
            f"edge_operator 必须是 {EDGE_OPERATORS} 之一", code="E101", got=cfg.edge_operator
        )
    if cfg.classifier not in ("auto", "sklearn", "numpy"):
        raise ValidationError("classifier 必须是 auto / sklearn / numpy", code="E101")
    if cfg.classifier_c <= 0.0:
        raise ValidationError("classifier_c 必须 > 0", code="E101", got=cfg.classifier_c)
    if cfg.max_iter < 1:
        raise ValidationError("max_iter 必须 >= 1", code="E101", got=cfg.max_iter)

    if cfg.hpo_trials < 1:
        raise ValidationError("hpo_trials 必须 >= 1", code="E101", got=cfg.hpo_trials)
    if not isinstance(cfg.tier1_only, bool):
        raise ValidationError("tier1_only 必须是 bool", code="E101")
    return cfg
