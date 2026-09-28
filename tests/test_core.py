"""core 层单测：错误码、配置、seed 派生、能力探测。"""

from __future__ import annotations

import numpy as np
import pytest

from graphforge.core import capabilities as caps
from graphforge.core.config import GraphForgeConfig, config_from_env, load_config
from graphforge.core.errors import (
    ConfigError,
    GraphForgeError,
    ValidationError,
)
from graphforge.core.seed import derive_seed, make_rng, normalize_seed, set_global_seed


def test_error_codes_are_stable():
    assert GraphForgeError("x").code == "E000"
    assert ValidationError("x").code == "E101"
    assert ConfigError("x").code == "E100"
    message = str(ValidationError("bad", code="E101", got=3))
    assert "E101" in message and "got=3" in message


def test_config_defaults_validate():
    cfg = load_config()
    assert cfg.seed == 42
    assert cfg.dim == 64
    assert abs(cfg.train_ratio + cfg.valid_ratio + cfg.test_ratio - 1.0) < 1e-12


def test_env_override(monkeypatch):
    monkeypatch.setenv("GRAPHFORGE_SEED", "7")
    monkeypatch.setenv("GRAPHFORGE_DIM", "32")
    monkeypatch.setenv("GRAPHFORGE_TIER1_ONLY", "true")
    cfg = config_from_env(GraphForgeConfig())
    assert cfg.seed == 7
    assert cfg.dim == 32
    assert cfg.tier1_only is True


def test_env_override_bad_value(monkeypatch):
    monkeypatch.setenv("GRAPHFORGE_DIM", "not-an-int")
    with pytest.raises(ConfigError) as exc:
        config_from_env(GraphForgeConfig())
    assert "E102" in str(exc.value) or "E100" in str(exc.value)


def test_load_config_rejects_unknown_key():
    with pytest.raises(ConfigError) as exc:
        load_config(no_such_field=1)
    assert "E103" in str(exc.value)


@pytest.mark.parametrize(
    "overrides",
    [
        {"dim": 1},
        {"walk_length": 1},
        {"num_walks": 0},
        {"p": 0.0},
        {"q": -1.0},
        {"epochs": 0},
        {"learning_rate": 0.0},
        {"batch_size": 8},
        {"optimizer": "rmsprop"},
        {"edge_operator": "magic"},
        {"classifier": "xgboost"},
        {"graph": "magic"},
        {"n_nodes": 1},
        {"test_ratio": 0.0},
        {"hpo_trials": 0},
    ],
)
def test_validation_rejects_bad_values(overrides):
    with pytest.raises(ValidationError) as exc:
        load_config(**overrides)
    assert "E101" in str(exc.value)


def test_ratio_sum_must_be_one():
    with pytest.raises(ValidationError):
        load_config(train_ratio=0.5, valid_ratio=0.1, test_ratio=0.2)


def test_seed_derivation_is_deterministic_and_distinct():
    assert derive_seed(42, "walk") == derive_seed(42, "walk")
    assert derive_seed(42, "walk") != derive_seed(42, "embed")
    assert derive_seed(42, "walk") != derive_seed(43, "walk")
    assert 0 <= derive_seed(-5, "x") <= 2**32 - 1
    assert normalize_seed(2**33 + 7) == 7


def test_rng_streams_are_reproducible():
    set_global_seed(123)
    a = make_rng(derive_seed(123, "walk")).random(5)
    b = make_rng(derive_seed(123, "walk")).random(5)
    assert np.array_equal(a, b)
    c = make_rng(derive_seed(124, "walk")).random(5)
    assert not np.array_equal(a, c)


def test_capability_tier_toggle():
    assert caps.backend_tier(False) in ("tier1", "tier2")
    assert caps.backend_tier(True) == "tier1"
    report = caps.capability_report(True)
    assert report["tier"] == "tier1"
    assert report["sklearn"] is False


def test_forced_tier1_context_restores_state():
    before = caps.force_tier1_enabled()
    with caps.forced_tier1():
        assert caps.force_tier1_enabled() is True
        assert caps.available_sklearn() is False
    assert caps.force_tier1_enabled() == before


def test_capability_probe_never_raises():
    assert caps.available_numpy() is True
    assert isinstance(caps.available_gensim(), bool)
    assert isinstance(caps.available_torch(), bool)
